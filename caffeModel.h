#ifndef CAFFEMODEL_H
#define CAFFEMODEL_H
#include "caffe_dnn.h"
#include <queue>

#define TARGET_CLASS_STRING "Pizza"
#define NUM_CLASSES 2
#define WAIT_KEY_DELAY 1  // '0' means waiting for a key press
#define ENABLE_PATCH_DISTRIBUTION false


using namespace cv;
using namespace std;

class caffeModel
{
	public:
        caffeModel(void);
        double guess(Mat, int); // Image, maximum batch size for GPU, visualize detected positive pizza patches
        void initialize(void); // set classifier using paths
		void initialize(std::string); // set classifier and load setting from a config file
        void set_mode(string); // GPU or CPU selection switch
        std::vector<Prediction> predict(Mat); // predict for a given image for all labels
        std::vector< std::vector<Prediction> > batchPredict(vector<cv::Mat>& images, int num_classes, int batch_size);
        Prediction predict(Mat, int); // predict for a given image for given label
        std::vector< float> batchPredictTarget(vector<cv::Mat>& images, int new_batch_size);
		void enableLogging(bool , std::string );

	public:
        std::string MODEL_PATH;
        std::string MEAN_PATH;
        std::string LABEL_PATH;
        std::string TRAINED_PATH;

        double TARGET_PROBABILITY = 0.80;
        double REQUIRED_PERCENTAGE = 0.00;
        int BATCH_SIZE = 80;
        int TARGET_CLASS = 963;
		int BUFFER_SIZE = 10;
		int SLIDING_WINDOW_LENGTH_MINIMUM = 100;
		double SLIDING_WINDOW_MAX_LENGTH_RATIO = 0.99;		
		double SLIDING_WINDOW_STEP_RATIO = 0.44;
		double SLIDING_WINDOW_SHRINKING_RATIO = 0.3;
        size_t window_length = 100, window_stride = 100;
        size_t window_enlarging_const = 100;
        Classifier* classifier;
		double operator<<(cv::Mat);
		

	private:
		int bufferCounter  = 0; // For sequence analysis
		int successCounter = 0; // For sequence analysis
		std::queue<bool> buffer; // Store previous results
};

#endif // CAFFEMODEL_H
