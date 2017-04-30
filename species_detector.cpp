#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdio.h>

// MACRO
#define GORI "gorilla"
#define CHIM "chimpanzee"
#define BACK "background"
#define GORI_VS_ALL 0
#define CHIM_VS_ALL 1
#define BACK_VS_ALL 2
#define POS 0
#define NEG 1

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
//using std::string;
using namespace std;
using namespace cv;

// ======== TYPE DEFINITIONS
typedef pair<string, float> Prediction; /* Pair (label, confidence) representing a prediction. */
typedef pair<Rect, vector<Prediction> > Detection; /* Rect and the prediction vector associated to it */

// GLOBAL VARIABLES
float th_classification = 0.106; // computed from ROC curve, used in final detectiion to reject or draw a bounding box
int threshold_value = 128;
int threshold_type = THRESH_BINARY;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
int K = 4; // number of clusters
Scalar colorTab[] = { Scalar(0, 0, 255), Scalar(0,255,0), Scalar(255,100,100), Scalar(255,0,255), Scalar(0,255,255) };
//float th_detection = 0.5; // threshold used in detection to reject or draw a region

class Classifier {
public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);
  
  std::vector<Prediction> Classify(const cv::Mat& img, int N);
  
private:
  void SetMean(const string& mean_file);
  
  std::vector<float> Predict(const cv::Mat& img);
  
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  
  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
  
private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  
public:
  int nClasses;
};

// creates a contingency table
vector<int> contingency_table(Classifier classifier, string testset_file, float threshold);

// creates a binary contingency table
vector<int> binary_contingency_table(Classifier classifier, string testfile, float threshold); 

// get probability predictions of all testset and dump to a file
vector< vector<Prediction> > getProbMatrix(Classifier classifier, ifstream& file, string dir);

// get a contingency table for each possible threshold [0,1], consider delta-t
vector< vector<int> > getContingencyTables(vector<vector<Prediction> >& probMatrix, ifstream& file, int mode);

// find probability assigned to class c
float findPrediction(vector<Prediction> predictions, string c);

// dumps to a file. FPRs in the first line and TPRs in the second line
void dump_fpr_tpr(vector<vector<int> > contTables, string filename);

// get weighted accuracy form an n by n contingency table
float get_weighted_accuracy(vector<int> table_nbyn, int nClasses);

// append accuracy for a particular epoch to a file
void dump_accuracy(string filename, float w_accuracy);

// average k-fold cross-validation. [just one classifier per time]
// appends the scores of each instance in a fold to a file. Then all the scores are used to produce the ROC.
void dump_scores(Classifier classifier, ifstream& file, string positiveClass, string dir);

void dump_score_vectors(Classifier classifier, ifstream& im_file, string dir);
 
// create contingency tables from a file with scores [used to average the folds]
vector<vector<int> > create_contTables(ifstream& labelfile, ifstream& scorefile, int mode);

// DETECTION FUNCTIONS
void test_proposal_f1_score_vs_iou(string directory);
vector<Detection>* region_classification(Mat* image, vector<Rect>* proposed_regions, Classifier classifier);
vector<Rect>* region_proposal(Mat* image); // <-- this function carries the complete detection pipeline using those below
vector<pair<Rect, string> >* region_labels(vector<Detection>* c_regions);
void draw_boxes_and_labels(Mat* final_image, vector<pair<Rect,string> >* max_regions);
vector<Rect> proposal(Mat* image);
Mat draw_boxes(Mat* image, vector<Rect> regions);
int count_pixel(Mat* image, Rect* window, int pixel_value);
vector<Point2f> get_centers(vector<Rect> regions);
Mat draw_points(Mat* image, vector<Point2f> points);
void draw_clusters(Mat* cluster_image, vector<Point2f>* region_centers, Mat* labels, vector<Point2f>* centers);
void get_centroids(Mat* centers, vector<Point2f>* centroids, int k);
void find_regions_from_clusters(vector<Point2f>* centroids, vector<Point2f>* points, Mat* labels, int K, vector<Rect>* cnn_regions);
void draw_boxes(Mat* image, vector<Rect>* regions);
void draw_boxes(Mat* image,  vector<Rect>* boxes, Scalar color);
int threshold_selection(Mat* image);

// UTILITY FUNCTIONS
pair<Mat, vector<Rect>* >* init_from_gt_file(string file_i); // from a file, return the image and the ground-truth boxes
vector<pair<float, float> >* compute_F1_vs_IoU(vector<Rect>* gt_boxes, vector<Rect>* p_regions); // returns vector of (IoU, F1-score) points
int areaOfIntersection(Rect A, Rect B);
vector<pair< float, float> >* average_f1(vector<vector<pair<float, float> >* >* all_IoU_F1); // compute overall pairs of (IoU,F1-score)
void dump_IoU_F1(vector<pair<float, float> >* overall_IoU_F1);

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
  
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  
  /* Load the binaryproto mean file. */
  SetMean(mean_file);
  
  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));
  
  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels()) << "Number of labels is different from the output layer dimension.";
  
  nClasses = labels_.size();
}

static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }
  
  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  string file = argv[5];
  
  std::cout << "---------- Prediction for " << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  int N = 5; // top N predictions
  std::vector<Prediction> predictions = classifier.Classify(img, N);
  
  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;    
  }

  // DETECTION
  namedWindow("after region proposal", 2);
  vector<Rect>* p_regions = region_proposal(&img);
  Mat cnn_regions = img.clone();
  draw_boxes(&cnn_regions, p_regions);
  //imshow("after region proposal", cnn_regions); // visualize regions
  vector<Detection>* c_regions = region_classification(&img, p_regions, classifier);
  
  // 1. function that returns regions with high probabilities
  // 2. function that does region merging
  // 3. draw boxes

  namedWindow("after cnn detection", 1);
  Mat final_image = img.clone();
  vector<pair<Rect, string> >* max_regions = region_labels(c_regions);
  draw_boxes_and_labels(&final_image, max_regions);
  //imshow("after cnn detection", final_image);

  // waitKey(0);
  string dir_test_proposal = "/home/carlo/bristoluni/short_individual_project/ground_truth/";
  test_proposal_f1_score_vs_iou(dir_test_proposal);
  
#if 0
  float lower_threshold = 0.0;
  float upper_threshold = 1.0;
  
  for(float i = lower_threshold; i <= upper_threshold; i += 0.05){
    printf("t = %.2f: ", i);
    // compute the contingency table of all test set
    string testset_file = "./data/mydata/test_0.txt";
    vector<int> contingencyTable = contingency_table(classifier, testset_file, i);
    // print table
    for(int i = 0; i < contingencyTable.size(); i++){
      printf("%d ", contingencyTable[i]);
    }
    printf("\n");
  }
#endif
  
#if 0
  float lower_threshold = 0.001;
  float upper_threshold = 0.002;
  for(float i = lower_threshold; i <= upper_threshold; i += 0.0001){

    printf("contingency table, threshold = %f: ", i);
    // compute contingency table of Chimpanzee vs Background
    string testfile = "./data/mydata/chimp_vs_back.txt";
    vector<int> table = binary_contingency_table(classifier, testfile, i);
    // print table
    for(int j = 0; j < table.size(); j++){
      printf("%d ", table[j]);
    }
    printf("\n");
    
  }
#endif
  
  string fold = "1";
  string dir =          "./data/mydata/training/"; // test on the cropped images
  string testset_file = "./data/mydata/test_" + fold + ".txt";
  //string testset_file = "./data/mydata/test_0123.txt";
  
  // open file
  ifstream file_test;
  file_test.open(testset_file.c_str());
  
#if 0
  // dump the vector of scores and the labels in two separated files, then used to compute MSE
  dump_score_vectors(classifier, file_test, dir);
#endif


#if 0
  //file_test.open(testset_file.c_str());
  vector<vector<Prediction> > probMatrix =  getProbMatrix(classifier, file_test, dir);
  vector<vector<int> > contTables =  getContingencyTables(probMatrix, file_test, CHIM_VS_ALL);
  dump_fpr_tpr(contTables, "chim_vs_all_" + fold + ".txt" );
  //dump_fpr_tpr(contTables, "chim_vs_all_entireset.txt" );
#endif
  
#if 0
  // append to file the scores assigned to each instance in a fold
  //file_test.open(testset_file.c_str());
  dump_scores(classifier, file_test, CHIM, dir);
  file_test.close();
#endif  

#if 0
  //file_test.open(testset_file.c_str());
  ifstream scorefile;
  scorefile.open("./all_scores_finetuned_last.txt"); 
  vector<vector<int> > contTables =  create_contTables(file_test, scorefile, CHIM_VS_ALL);
  dump_fpr_tpr(contTables, "chim_vs_all_0123_finetuned_last.txt" );
#endif
  
#if 0
  vector<int> table_nbyn = contingency_table(classifier, testset_file, 0.5);
  float w_accuracy= get_weighted_accuracy(table_nbyn, classifier.nClasses);
  // append accuracy to file
  dump_accuracy("./myexperiments/train_accuracy_vs_epochs.txt", w_accuracy);
#endif

  cout << "done" << endl;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV


vector<int> contingency_table(Classifier classifier, string testset_file, float threshold){
  string imagename;	// image to process
  int label;			// true label associated to the image
  string dir = "./data/mydata/training/"; // test on the cropped images
  int nclasses = classifier.nClasses;
  vector<int> table(nclasses * nclasses, 0); // initialize table with all zeros
  
  ifstream file;
  file.open(testset_file.c_str());
  
  std::string word;
  while (file >> word){
    // read image name and label from file
    imagename = dir + word;
    file >> word;
    label = atoi(word.c_str());
    
    // open image
    cv::Mat img = cv::imread( imagename, -1);
    CHECK(!img.empty()) << "Unable to decode image " << imagename;
    // classify and get prediction top N=nClasses predictions
    std::vector<Prediction> predictions = classifier.Classify(img, nclasses);
    
    //update contingency table
    string p = predictions[0].first;
    int pred;
    if (p.find(CHIM) != std::string::npos) {
      pred = 1;
    } else if(p.find(GORI) != std::string::npos){
      pred = 0;
    } else 
      pred = 2;
    // update contingency table
    table[label * nclasses + pred]++;   	
  }
  
  return table;
}

vector<int> binary_contingency_table(Classifier classifier, string testfile, float threshold){
  string imagename;	// image to process
  int label;		// true label associated to the image
  string dir = "./data/mydata/training/"; // test on the cropped images
  int nclasses = 2;
  vector<int> table(nclasses * nclasses, 0); // initialize table with all zeros
  
  ifstream file;
  // open file
  file.open(testfile.c_str());
  
  std::string word;
  while (file >> word){
    // read image name and label from file
    imagename = dir + word;
    file >> word;
    label = atoi(word.c_str());
    
    // open image
    cv::Mat img = cv::imread( imagename, -1);
    CHECK(!img.empty()) << "Unable to decode image " << imagename;
    // classify and get all predictions
    std::vector<Prediction> predictions = classifier.Classify(img, classifier.nClasses);
    
    float confidence = 0.0;
    // find chimpanzee prediction
    for(int i = 0; i < predictions.size(); i++){
      string pred_label = predictions[i].first;
      if (pred_label.find("troglodytes") != std::string::npos){
	confidence = predictions[i].second;
	break;
      }
    }
    
    int pred;
    if(confidence >= threshold){
      // predict chimpanzee
      pred = 0;
    } else {
      // predict background
      pred = 1;
    }
    
    // update contingency table
    table[label * nclasses + pred]++;   	
  }
  
  return table;
}

vector< vector<Prediction> > getProbMatrix(Classifier classifier, ifstream& file, string dir){
  vector< vector<Prediction> > probMatrix;
  
  string word;
  while (file >> word){
    // read image name and label from file
    string imagename = dir + word;
    file >> word; // waste one word for the lable
    
    // open image
    cv::Mat img = cv::imread( imagename, -1);
    CHECK(!img.empty()) << "Unable to decode image " << imagename;
    
    // get all pair predictions (label, prob)
    std::vector<Prediction> predictions = classifier.Classify(img, classifier.nClasses);  
    
    // push prediction vector into the matrix
    probMatrix.push_back(predictions);
  }
  //shrink vector to fit its data
  probMatrix.resize(probMatrix.size());
  return probMatrix;
}

vector<vector<int> > getContingencyTables(vector<vector<Prediction> >& probMatrix, ifstream& file, int mode){
  vector<int> labelvector;
  
  // initialize label vector
  file.clear();
  file.seekg(0, ios::beg);
  string word;
  while (file >> word){
    // waste one word for the filename, the second is the label
    file >> word;
    labelvector.push_back(atoi(word.c_str()));
  }
  // shrink vector just to fit data
  labelvector.resize(labelvector.size());

  float lower_threshold = 0.0;
  float upper_threshold = 1.0;
  float delta_t = 0.001;
  int nThresh = 1000; //(upper_threshold - lower_threshold) / delta_t;

  // initialize an empty vector of binary contingency tables, one for each threshold
  vector<vector<int> > contTables(nThresh, vector<int>(4,0));
  
  // for each image processed
  for(int i = 0; i < probMatrix.size(); i++){
    float p_chim = findPrediction(probMatrix[i], CHIM);
    float p_gori = findPrediction(probMatrix[i], GORI);
    float p_back = findPrediction(probMatrix[i], BACK);
    int index = 0;
    // for each threshold
    for(float t = lower_threshold; t < upper_threshold - delta_t; t += delta_t){ 
      int pred;
      switch(mode){
      case CHIM_VS_ALL:
	pred = p_chim >= t ? POS : NEG;
	break;
      case GORI_VS_ALL:
	pred = p_gori >= t ? POS : NEG;
	break;
      case BACK_VS_ALL:
	pred = p_back >= t ? POS : NEG;
	break;
      default:
	fprintf(stderr, "none applies to mode");
	pred = NEG;
	break;
      }
      // update table 
      int label = labelvector[i] == mode ? POS : NEG;
      contTables[index][label * 2 + pred]++;
      index++;
    } 
  }
  return contTables;
}

float findPrediction(vector<Prediction> predictions, string c){
  float confidence = 0.0;
  // find probability assigned to class c
  for(int i = 0; i < predictions.size(); i++){
    string pred_label = predictions[i].first;
    if (pred_label.find(c) != std::string::npos){
      confidence = predictions[i].second;
      break;
    }
  }
  return confidence;
}

void dump_fpr_tpr(vector<vector<int> > contTables, string filename){
  // the contingency table is stored in 2x2 flattened array like this: [TP, FN, FP, TN]
  int Pos = contTables[0][0] + contTables[0][1];
  int Neg = contTables[0][2] + contTables[0][3];
  string false_positive_rate = "";
  string true_positive_rate = "";
  
  for(int i  = 0; i < contTables.size(); i++){
    vector<int> table = contTables[i];
    int TP = table[0];
    int FP = table[2];
    float TPR = (float) TP / Pos;
    float FPR = (float) FP / Neg;
    false_positive_rate += std::to_string(FPR) + ' ';
    true_positive_rate  += std::to_string(TPR) + ' ';
  } 

  // erase the last ' ' from both strings
  false_positive_rate.erase(false_positive_rate.size() - 1);
  true_positive_rate.erase(true_positive_rate.size() - 1);
  // write into the file
  ofstream outfile;
  outfile.open("./myexperiments/finetuned_last/" + filename);
  outfile << false_positive_rate << endl;
  outfile << true_positive_rate << endl;
  //close file
  outfile.close();
}

float  get_weighted_accuracy(vector<int> table_nbyn, int nClasses){
  vector<int> classMarginals(nClasses, 0);
  vector<int> diagonal(nClasses);
  int total = 0; // total number of instances

  // get per-class marginals
  for(int i = 0; i < nClasses; i++)
    for(int j = 0; j < nClasses; j++)
      classMarginals[i] += table_nbyn[i * nClasses + j];

  //compute total number of instances
  for(int i = 0; i < classMarginals.size(); i++)
    total += classMarginals[i];

  // get diagonal of the contingency table
  for(int i = 0 ; i < nClasses; i++)
    diagonal[i] = table_nbyn[i * nClasses + i];

  float accuracy = 0;
  // compute weighted accuracy
  for(int i = 0; i < classMarginals.size(); i++)
    accuracy += (diagonal[i] / (float)classMarginals[i])  * (classMarginals[i] / (float) total);
  
  return accuracy;
}

void dump_accuracy(string filename, float accuracy){
  ofstream outfile;
  outfile.open(filename, std::ios_base::app);
  outfile << accuracy << endl;
  outfile.close();
}

void dump_scores(Classifier classifier, ifstream& file, string positiveClass, string dir){
  string word;
  ofstream outfile;
  outfile.open("all_scores_finetuned_last.txt", std::ios_base::app);

  while (file >> word){
    // read image name and label from file
    string imagename = dir + word;
    file >> word; // waste one word for the label
    
    // open image
    cv::Mat img = cv::imread( imagename, -1);
    CHECK(!img.empty()) << "Unable to decode image " << imagename;
    
    // get all pair predictions (label, prob)
    std::vector<Prediction> predictions = classifier.Classify(img, classifier.nClasses);  
    float confidence = findPrediction(predictions, positiveClass);
    outfile << std::to_string(confidence) << endl;
  }
  outfile.close();
}

vector<vector<int> > create_contTables(ifstream& labelfile, ifstream& scorefile, int mode){
  vector<int> labelvector;
  
  // initialize label vector
  labelfile.clear();
  labelfile.seekg(0, ios::beg);
  string word;
  while (labelfile >> word){
    // waste one word for the filename, the second is the label
    labelfile >> word;
    labelvector.push_back(atoi(word.c_str()));
  }
  
  float lower_threshold = 0.0;
  float upper_threshold = 1.0;
  float delta_t = 0.001;
  int nThresh = 1000; //(upper_threshold - lower_threshold) / delta_t;
  
  // initialize an empty vector of binary contingency tables, one for each threshold
  vector<vector<int> > contTables(nThresh, vector<int>(4,0));

  scorefile.clear();
  scorefile.seekg(0, ios::beg);

  // for each image processed
  string score;
  int i = 0;
  while(scorefile >> score){
    float probability = atof(score.c_str());
    int index = 0;
    // for each threshold
    for(float t = lower_threshold; t < upper_threshold - delta_t; t += delta_t){ 
      int pred = probability >= t ? POS : NEG;
      // update table 
      int label = labelvector[i] == mode ? POS : NEG;
      contTables[index][label * 2 + pred]++;
      index++;
    } 
    i++;
  }
  return contTables;
}

void dump_score_vectors(Classifier classifier, ifstream& im_file, string dir){
  string word;
  ofstream score_file;
  ofstream label_file;
  //score_file.open("score_vectors_reference.txt", std::ios_base::app);
  //label_file.open("label_vectors_reference.txt", std::ios_base::app);
  //score_file.open("score_vectors_finetuned_all.txt", std::ios_base::app);
  //label_file.open("label_vectors_finetuned_all.txt", std::ios_base::app);
  score_file.open("score_vectors_finetuned_last.txt", std::ios_base::app);
  label_file.open("label_vectors_finetuned_last.txt", std::ios_base::app);

  while (im_file >> word){
    // read image name and label from file
    string imagename = dir + word;
    im_file >> word; // read label
    
    // open image
    cv::Mat img = cv::imread( imagename, -1);
    CHECK(!img.empty()) << "Unable to decode image " << imagename;
    
    // get all pair predictions (label, prob)
    std::vector<Prediction> predictions = classifier.Classify(img, classifier.nClasses);  
    float p_0 = findPrediction(predictions, GORI); // probability of class i = 0,1,2
    float p_1 = findPrediction(predictions, CHIM);
    float p_2 = findPrediction(predictions, BACK);
    p_2 = (p_2 == 0.0) ? 1 - (p_0 + p_1) : p_2;  // to handle the fact that the reference does not have BACK class
    
    score_file << std::to_string(p_0) << " " << std::to_string(p_1) << " " << std::to_string(p_2) << endl;
    label_file << word << endl;
  }

  score_file.close();
  label_file.close();
}

/* ======== DETECTION FUNCTIONS ============= */
vector<Rect>* region_proposal(Mat* image){
  Mat image_gray, th_image, labels, centers;
  // Convert the image to Gray
  cvtColor(*image, image_gray, CV_BGR2GRAY );
  // threshold
  //threshold_value = threshold_selection(&image_gray);
  threshold( image_gray, th_image, threshold_value, max_BINARY_value, threshold_type );
  // window proposal
  vector<Rect> regions = proposal(&th_image);
  // window centers
  vector<Point2f> points = get_centers(regions);
  // clustering
  kmeans(points, K, labels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 100, 1), 5, KMEANS_PP_CENTERS, centers);
  // get centroids
  vector<Point2f>* centroids = new vector<Point2f>;
  get_centroids(&centers, centroids, K);
  // get cnn regions
  vector<Rect>* cnn_regions = new vector<Rect>;
  find_regions_from_clusters(centroids, &points, &labels, K, cnn_regions);
  return cnn_regions;
}

vector<Rect> proposal(Mat* image){
  vector<Rect> regions = vector<Rect>();
  // Parameters of your sliding window
  int kernel_rows = image->rows / 20, kernel_cols = image->cols / 20, stridex = kernel_cols / 2, stridey = kernel_rows / 2; // 10, 10, 20
  // acceptance threshold
  float threshold = 0.9; // 80% of the region should be black
  
  for (int row = 0; row <= image->rows - kernel_rows; row += stridey){
    for (int col = 0; col <= image->cols - kernel_cols; col += stridex){
      // resulting window
      Rect window(col, row, kernel_cols, kernel_rows);
      // feature evaluator over Windows
      int black_pixels = count_pixel(image, &window, 0);
      float percentage = black_pixels / float(kernel_rows * kernel_cols);
      if( percentage >= threshold)
	regions.push_back(window);
    }
  }
  return regions;
}

Mat draw_boxes(Mat* image, vector<Rect> regions){
  Mat outimage = image->clone();
  for(unsigned int i = 0; i < regions.size(); i++){
    Rect window = regions[i];
    rectangle(outimage, window, Scalar(255), 5, 8, 0);
  }
  return outimage;
}

int count_pixel(Mat* image, Rect* window, int pixel_value){
  int count = 0;
  Mat temp = (*image)(*window);
  for(int i  = 0; i < temp.rows; i++)
    for(int j = 0; j < temp.cols; j++)
      if(temp.at<uchar>(i,j) == 0)
	count++;
  return count;
}

vector<Point2f> get_centers(vector<Rect> regions){
  vector<Point2f> points = vector<Point2f>(regions.size());
  for(unsigned int i = 0; i < regions.size(); i++){
    int xc = regions[i].x + (regions[i].width / 2);
    int yc = regions[i].y + (regions[i].height / 2);
    points.push_back(Point(xc,yc));
  }
  return points;
}

Mat draw_points(Mat* image, vector<Point2f> points){
  Mat outimage = image->clone();
  for(unsigned int i = 0; i < points.size(); i++){
    Point p = points[i];
    circle( outimage, p, 10, Scalar(255,100,100), -1, 8 );
  }
  return outimage;
}

void draw_clusters(Mat* cluster_image, vector<Point2f>* region_centers, Mat* labels, vector<Point2f>* centers){
  // draw points in each cluster
  for(unsigned int i = 0; i < region_centers->size(); i++ ){
    int clusterIdx = labels->at<int>(i);
    Point ipt = (*region_centers)[i];
    circle( *cluster_image, ipt, 10, colorTab[clusterIdx], -1, 8);
  }

  // draw centroids
  for(unsigned int i = 0; i < centers->size(); i++){
    circle(*cluster_image, (*centers)[i] , 15, Scalar(255,255,255), -1, 8);
  }
}

void get_centroids(Mat* centers, vector<Point2f>* centroids, int k){
  for(int i = 0; i < centers->rows; i++) {
    Point2f pt = centers->at<Point2f>(i);
    centroids->push_back(pt);
  }
}

void find_regions_from_clusters(vector<Point2f>* centroids, vector<Point2f>* points, Mat* labels, int K, vector<Rect>* cnn_regions){
  // for each cluster find min_x, min_y, max_x, max_y
  // use array[4] and each time you encounter a point of that cluster update mins and maxs
  // create rectangle Rect(min_x, min_y, max_x - min_x, max_y - min_y)
  // push Rect to cnn_regions
  vector<vector<int> > c_extremes(K, vector<int>(4));
  
  // initialize mins = MAX_INT, max = 0
  for(unsigned int i = 0; i < c_extremes.size(); i++){
    c_extremes[i][0] = c_extremes[i][1] = std::numeric_limits<int>::max();
    c_extremes[i][2] = c_extremes[i][3] = std::numeric_limits<int>::min();
  }
  
  for(unsigned int i = 0; i < points->size(); i++){
    int cluster_id = labels->at<int>(i);
    Point2f p = (*points)[i];
    // update min_x, min_y, max_x, max_y
    c_extremes[cluster_id][0] = c_extremes[cluster_id][0] <= p.x ? c_extremes[cluster_id][0] : p.x;
    c_extremes[cluster_id][1] = c_extremes[cluster_id][1] <= p.y ? c_extremes[cluster_id][1] : p.y;
    c_extremes[cluster_id][2] = c_extremes[cluster_id][2] > p.x ? c_extremes[cluster_id][2] : p.x;
    c_extremes[cluster_id][3] = c_extremes[cluster_id][3] > p.y ? c_extremes[cluster_id][3] : p.y;
  }
  
  for(int i = 0; i < K; i++){
    int x = c_extremes[i][0];
    int y = c_extremes[i][1];
    int width = c_extremes[i][2] - x;
    int height = c_extremes[i][3] - y;
    cnn_regions->push_back(Rect(x, y, width, height));
    cout << x << "," << y << " " << x + width << "," << y + height << endl;
  }
}

void draw_boxes(Mat* image, vector<Rect>* regions){
  for(unsigned int i = 0; i < regions->size(); i++)
    rectangle(*image, (*regions)[i], colorTab[i], 10);
}

vector<Detection>* region_classification(Mat* image, vector<Rect>* proposed_regions, Classifier classifier){
  vector<Detection>* d_regions = new vector<Detection>;
  for(unsigned int i = 0; i < proposed_regions->size(); i++){
    Rect region = (*proposed_regions)[i];
    Mat cropped_image = (*image)(region);
    vector<Prediction> predictions = classifier.Classify(cropped_image, classifier.nClasses);
    float p_gori = findPrediction(predictions, GORI);
    float p_chim = findPrediction(predictions, CHIM);
    float p_back = 1 - (p_gori + p_chim);
    Prediction pred_gori(GORI, p_gori);
    Prediction pred_chim(CHIM, p_chim);
    Prediction pred_back(BACK, p_back);
    vector<Prediction> region_pred = {pred_gori, pred_chim, pred_back};
    d_regions->push_back(Detection(region, region_pred));
  }
  return d_regions;
}

vector<pair<Rect, string> >* region_labels(vector<Detection>* c_regions){
  vector<pair<Rect, string> >* max_regions = new vector<pair<Rect, string> >;
  for(unsigned int i = 0; i < c_regions->size(); i++){
    Rect r = (*c_regions)[i].first;
    vector<Prediction> p = (*c_regions)[i].second;
    float p_gori = p[0].second;
    float p_chim = p[1].second;
    printf("gori: %.3f, chim: %.3f\n", p_gori, p_chim);
    float max_score = p_gori >= p_chim ? p_gori : p_chim;
    int max_pred = p_gori >= p_chim ? 0 : 1;
    string label = max_pred == 0 ? GORI : CHIM;
    if(max_score >= th_classification){
      char s[10];
      sprintf(s, "%.2f", max_score);
      max_regions->push_back(pair<Rect, string>(r,label + "-" + string(s)));   
    }
  }
  return max_regions;
}

void draw_boxes_and_labels(Mat* image, vector<pair<Rect,string> >* max_regions){
  for(unsigned int i = 0; i < max_regions->size(); i++){
    Rect r = (*max_regions)[i].first;
    string label = (*max_regions)[i].second;
    rectangle(*image, r, colorTab[i % 5], 5);
    putText(*image, label, Point(r.x + 30, r.y + 40), FONT_HERSHEY_PLAIN, 4 , Scalar(0,255,255), 2, CV_AA);
  }
}

// UTILITY FUNCTION IMPLEMENTATIONS

pair<Mat, vector<Rect>* >* init_from_gt_file(string file_i){
  int n; // number of objects
  string image_name;
  
  ifstream f;
  f.open(file_i.c_str());
  if( !f ){
    fprintf(stderr, "error reading file\n");
  }

  // 1. read the image path, then create image
  f >> image_name;
  cout << image_name << endl;
  Mat img = cv::imread(image_name, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file_i;

  // 2. read the number of objects
  f >> n;
  vector<Rect>* gt_boxes = new vector<Rect>;
  // read the objects coordinates
  for(int i = 0; i < n; i++){
    int x1, y1, x2, y2, label; // discard label
    f >> label >> x1 >> y1 >> x2 >> y2;
    gt_boxes->push_back(Rect(Point(x1,y1), Point(x2,y2)));
  }
  
  f.close();
  return new pair<Mat, vector<Rect>* >(img, gt_boxes);
}

vector<pair<float, float> >* compute_F1_vs_IoU(vector<Rect>* gt_boxes, vector<Rect>* p_regions){
  if( gt_boxes->size() == 0){
    fprintf(stderr, "label size is 0");
  }

  vector<pair<float, float> >* IoU_F1 = new vector<pair<float, float> >;
  int* checked = (int*) calloc(gt_boxes->size(), sizeof(int)); // how many times each ground-truth box is detected
  float* IoU = (float*) calloc(gt_boxes->size() * p_regions->size(), sizeof(float)); // pairwise IoU
  float start_iou = 0.001;
  float end_iou = 1.0;
  float delta_iou = 0.001; 
  
  // compute pairwise intersection over union
  for(unsigned int i = 0; i < p_regions->size(); i++){
    Rect A = (*p_regions)[i];
    for( unsigned int j = 0; j < gt_boxes->size(); j++){
      Rect B = (*gt_boxes)[j];
      // compute the area of A intersecate B
      int areaI = areaOfIntersection(A,B);
      // compute the area of A union B = area(A) + area(B) - area(itersection)
      int areaU = (A.width * A.height) + (B.width * B.height) - areaI;
      
      // compute ratio  [ 100% in case of perfect overlap, down to 0% ]
      IoU[i * gt_boxes->size() + j] = (float) areaI / (float) areaU;
      printf("iou: %.3f\n",  (float) areaI / (float) areaU);
    }
  }
  
  // compute a pair of (IoU, F1-score) for each IoU threshold
  for(float t = start_iou; t <= end_iou; t += delta_iou){ 
    int TP = 0, FP = 0, FN = 0; // true positives, false positives, false negatives;
    for(unsigned int i = 0; i < p_regions->size(); i++)
      for( unsigned int j = 0; j < gt_boxes->size(); j++)
	// if the IoU is greather than the threshold, increment the number of times the ground-truth box is detected (TP)
	if(IoU[i * gt_boxes->size() + j] >= t)
	  checked[j]++;
      
    // compute TP, FN and FN 
    for(unsigned int i = 0; i < gt_boxes->size(); i++){
      TP += checked[i];
      FN = checked[i] == 0 ? FN + 1 : FN;
      // zero the checked position to set up the next iteration
      checked[i] = 0;
    }
    FP = p_regions->size() * gt_boxes->size() - TP;
    //cout << "TP, FP, FN: " << to_string(TP) << " " << to_string(FP) << " " << to_string(FN) << endl;
    if(FP < 0 || FN < 0){
      cout << "error" << endl;
      exit(1);
    }
    float f1_score = (2 * TP) / (float) (2 * TP + FN + FP);
    if(f1_score > 1 || f1_score < 0){
      printf("f1_score: %.3f\n", f1_score);
      exit(1);
    }
    IoU_F1->push_back(pair<float,float>(t, f1_score));
  }
  free(checked);
  free(IoU);
  return IoU_F1;
}

void test_proposal_f1_score_vs_iou(string directory){
  int testset_size = 30;
  vector<vector<pair<float, float> >* >* all_IoU_F1 = new vector<vector<pair<float, float> >* >;
  // for each image in the test set
  for(int i = 1; i <= testset_size; i++){
    string file_i = directory + std::to_string(i) + ".txt";
    // get image_path and ground-truth boxes
    pair<Mat, vector<Rect>* >* image_and_gt_boxes = init_from_gt_file(file_i);
    // get proposed regions
    Mat img = image_and_gt_boxes->first;
    vector<Rect>* p_regions = region_proposal(&img);

    // get IoU and F1-scores for each IOU threshold, for the current image
    vector<Rect>* gt_boxes = image_and_gt_boxes->second;
    vector<pair<float, float> >* IoU_F1 = compute_F1_vs_IoU(gt_boxes, p_regions);   
    all_IoU_F1->push_back(IoU_F1);

#if 1
    // write images
    string dir = "/home/carlo/bristoluni/short_individual_project/both_boxes/";
    string det = "best_threshold/";
    draw_boxes(&img,  gt_boxes, Scalar(0,255,0));
    draw_boxes(&img,  p_regions, Scalar(0,0,255));
    imwrite(dir + det + to_string(i) + ".jpg", img);
#endif

  }

  // compute overall f1-score 
  vector<pair< float, float> >* overall_IoU_F1 = average_f1(all_IoU_F1); 
  // first row IoU, second row F1-scores  
  dump_IoU_F1(overall_IoU_F1);
}

int areaOfIntersection(Rect A, Rect B){
  return max(0, min(A.x + A.width, B.x + B.width) - max(A.x, B.x)) * 
    max(0, min(A.y + A.height, B.y + B.height) - max(A.y, B.y) );
}

vector<pair< float, float> >* average_f1(vector<vector<pair<float, float> >* >* all_IoU_F1){
  int n = all_IoU_F1->size();
  int m = (*all_IoU_F1)[0]->size();
  vector<pair<float, float> >* overall_IoU_F1 = new vector<pair<float, float> >(m);
  // initialize
  for(int i = 0; i < m; i++){
    (*overall_IoU_F1)[i].first = 0.0;
    (*overall_IoU_F1)[i].second = 0.0;
  }

  for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++){
      (*overall_IoU_F1)[j].first = (*(*all_IoU_F1)[i])[j].first;
      (*overall_IoU_F1)[j].second += (*(*all_IoU_F1)[i])[j].second;
    }
  // average
  for(int i = 0; i < m; i++)
    (*overall_IoU_F1)[i].second /= (float) n;

  return overall_IoU_F1;
}


void dump_IoU_F1(vector<pair<float, float> >* overall_IoU_F1){
  string IoU = "";
  string f1_score = "";
  ofstream outfile;
  outfile.open("./myexperiments/proposals/iou_vs_f1scores_threshold.txt");
  int len = overall_IoU_F1->size();
  
  for(int i  = 0; i < len; i++){
    pair<float, float> p = (*overall_IoU_F1)[i];
    outfile << p.first;
    if(i < len-1)
      outfile << ' ';
    else
      outfile << endl;
  } 
  for(int i  = 0; i < len; i++){
    pair<float, float> p = (*overall_IoU_F1)[i];
    outfile << p.second;
    if(i < len-1)
      outfile << ' ';
    else
      outfile << endl;
  } 
 
  outfile << IoU << endl << f1_score << endl;
  //close file
  outfile.close(); 
}

void draw_boxes(Mat* image,  vector<Rect>* boxes, Scalar color){
  for(unsigned int i = 0; i < boxes->size(); i++){
    Rect r = (*boxes)[i];
    rectangle(*image, r, color, 5);
  }
}

int threshold_selection(Mat* image){
  int epsilon = 1, max_iter = 50, it = 0; // convergence criteria = abs(Ti - Ti+1) < epsilon
  int th_i = 128; // initial threshold
  bool converged = false;
  while(!converged && it < max_iter){
    //fprintf(stderr, "actual threshold: %d\n", th_i);
    // segment image
    vector<pair<Point, int> > g1;
    vector<pair<Point, int> > g2;
    for(int i = 0; i < image->rows; i++){
      for(int j = 0; j < image->cols; j++){
	int value = image->at<uchar>(j, i);
	if(value > th_i)
	  g1.push_back(pair<Point, int>(Point(j,i), value));
	else
	  g2.push_back(pair<Point, int>(Point(j,i), value));
      }
    }
    // compute m1 and m2
    float m1 = 0, m2 = 0;
    for(unsigned int i = 0; i < g1.size(); i++) m1 += g1[i].second;
    for(unsigned int i = 0; i < g2.size(); i++) m2 += g2[i].second;
    m1 /= g1.size(); m2 /= g2.size();
    int new_th = (m1 + m2) / 2;
    if(abs(th_i - new_th) < epsilon) converged = true;
    th_i = new_th;
    it++;
  }
  return th_i;
}
