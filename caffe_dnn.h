/*
 * caffe_dnn.h
 *
 * Implementation based on this example:
 * https://github.com/BVLC/caffe/tree/master/examples/cpp_classification
 *
 *  Created on: Jul 29, 2016
 *      Author: claudiu
 */

#ifndef CAFFE_DNN_H_
#define CAFFE_DNN_H_

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ctime>

using namespace caffe;
using std::string;

/*
 * Pair (label, confidence) representing a prediction.
 */
typedef std::pair<string, float> Prediction;

class Classifier
{
public:
	Classifier(const string& model_file,
			const string& trained_file,
			const string& mean_file,
			const string& label_file,
			const int TARGET_CLASS);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

	std::vector< vector<Prediction> > ClassifyBatch(const vector< cv::Mat > imgs, int num_classes = 5 , int new_batch_size=1);

	std::vector<float> ClassifyBatchTarget(const vector< cv::Mat > imgs, int new_batch_size);
	
	boost::shared_ptr<caffe::Net<float> > net_;
	
private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

	std::vector<float>  PredictBatch(const vector< cv::Mat > imgs, int new_batch_size);

	void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);

	void PreprocessBatch(const vector<cv::Mat> imgs,std::vector< std::vector<cv::Mat> >* input_batch);


private:
	cv::Size input_geometry_;
	int num_channels_;
	int TARGET_CLASS;
	cv::Mat mean_;
	std::vector<string> labels_;
};

#endif /* CAFFE_DNN_H_ */
