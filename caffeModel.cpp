#include "caffeModel.h" 
#include <fstream>

caffeModel::caffeModel(void)
{
    google::InitGoogleLogging("XXX");
    google::SetCommandLineOption("GLOG_minloglevel", "2");

    Caffe::set_mode(Caffe::GPU); // initially set for Gpu
}

void caffeModel::initialize(void)
{
    this->classifier =  new Classifier(MODEL_PATH, TRAINED_PATH, MEAN_PATH, LABEL_PATH,TARGET_CLASS);
}

pair< vector< double > , vector< string > > loadConfigurations(std::string CONFIG_FILE_PATH)
{
	ifstream configReader; configReader.open(CONFIG_FILE_PATH);
	string line;
	vector< double > parametersNumbers;
	vector< string > parametersStrings;
	//std::cout << "Here" << std::endl;
	for(int r = 0; r<6 ; r++)
	{
		getline(configReader, line);
		size_t pos = line.find_first_of("=");
		parametersNumbers.push_back(atof(line.substr(pos+2).c_str()));
		//cout << atof(line.substr(pos+1).c_str()) << endl; cin.get();
	}
	for (int r = 0; r<5; r++)
	{
		getline(configReader, line);
		size_t pos = line.find_first_of("=");
		parametersStrings.push_back(line.substr(pos+2));
		//cout << line.substr(pos+1) << endl; cin.get();
	}

	configReader.close();
	return std::make_pair(parametersNumbers, parametersStrings);
}

void caffeModel::initialize(std::string CONFIG_FILE_PATH)
{
	std::pair< std::vector< double > , std::vector< std::string > >  parameterList = loadConfigurations(CONFIG_FILE_PATH);

	this->SLIDING_WINDOW_LENGTH_MINIMUM		= parameterList.first[0];	// 100;
	this->SLIDING_WINDOW_MAX_LENGTH_RATIO	= parameterList.first[1];	// 0.99;
	this->SLIDING_WINDOW_STEP_RATIO			= parameterList.first[2];	// 0.44;
	this->SLIDING_WINDOW_SHRINKING_RATIO	= parameterList.first[3];	// 0.3;
	this->TARGET_PROBABILITY				= parameterList.first[4];   // 0.70
	this->REQUIRED_PERCENTAGE				= parameterList.first[5];   // 0

	this->MODEL_PATH						= parameterList.second[0];
	this->MEAN_PATH							= parameterList.second[1];
	this->LABEL_PATH						= parameterList.second[2];
	this->TRAINED_PATH						= parameterList.second[3];

	set_mode(parameterList.second[4]); // GPU

	this->classifier = new Classifier(MODEL_PATH, TRAINED_PATH, MEAN_PATH, LABEL_PATH, TARGET_CLASS);
}

void caffeModel::set_mode(string mode) // Choose gpu or cpu
{
    if(mode=="CPU")
    {
        Caffe::set_mode(Caffe::CPU);
    }
    else if(mode=="GPU")
    {
        Caffe::set_mode(Caffe::GPU);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }
}

std::vector< std::vector<Prediction> > caffeModel::batchPredict(vector<cv::Mat>& images, int num_classes, int batch_size)
{
    return this->classifier->ClassifyBatch(images,num_classes,batch_size);
}

std::vector< float> caffeModel::batchPredictTarget(vector<cv::Mat>& images, int new_batch_size)
{
    return this->classifier->ClassifyBatchTarget(images,new_batch_size);
}


std::vector<Prediction> caffeModel::predict(Mat image)
{
    CHECK(!image.empty()) << "Unable to decode image " <<endl;
	std::vector<Prediction> predictions = this->classifier->Classify(image,5);
	return predictions;
}

Prediction caffeModel::predict(Mat image,int label_number)
{
    //CHECK(!image.empty()) << "Unable to decode image " << endl;
	std::vector<Prediction> predictions = this->classifier->Classify(image,5);
	return predictions[label_number];
}

double caffeModel::guess(Mat REFERENCE, int batch_limit)
{
	//bool VISUALIZE_SWITCH = false; // Debugging pu

	Mat imageToDraw = REFERENCE; // Keep a copy of  the input to draw detected pizzas
	Mat bestPrediction;
	int massageHeight = imageToDraw.rows * 0.2; Mat resultMatrix(massageHeight, imageToDraw.cols, CV_8UC3, Scalar(180, 180, 180));

    size_t LENGTH = REFERENCE.cols;
    size_t HEIGHT = REFERENCE.rows;


	int box_length				= SLIDING_WINDOW_LENGTH_MINIMUM;	// std::max((int)(LENGTH * 0.2), 100);		// Minimum length
	int MAX_BOX_LENGTH			= LENGTH * SLIDING_WINDOW_MAX_LENGTH_RATIO;												// Starting max length
	int stride					= std::min(LENGTH * SLIDING_WINDOW_STEP_RATIO, HEIGHT * SLIDING_WINDOW_STEP_RATIO);		// Step size
	int WINDOW_ENLARGING_CONST	= LENGTH * SLIDING_WINDOW_SHRINKING_RATIO;												// Expanding const

    int window_length=min(LENGTH,HEIGHT);

    window_length = min(window_length,MAX_BOX_LENGTH); // Overiding default max window size

    int X_length=window_length-1,Y_length=window_length-1;


    std::vector<cv::Mat> patches;

    patches.push_back( REFERENCE );
	patches.push_back( REFERENCE( cv::Rect( 0			, 0			, LENGTH/2	, HEIGHT )));
	patches.push_back( REFERENCE( cv::Rect( LENGTH / 2	, 0			, LENGTH / 2, HEIGHT )));
	patches.push_back( REFERENCE( cv::Rect( LENGTH / 4	, HEIGHT / 4, LENGTH / 4, HEIGHT / 4 )));
	patches.push_back( REFERENCE( cv::Rect( LENGTH / 2	, HEIGHT / 2, LENGTH / 4, HEIGHT / 4 )));

   /* while(window_length>=box_length)  // Extracting patches from the image
    {
        Y_length=window_length-1;
        while(Y_length<HEIGHT) // vertical journey
        {
            X_length=window_length-1;
            while(X_length<LENGTH) // Horizontal journey
            {
                Rect image_patch(X_length-window_length+1,Y_length-window_length+1,window_length,window_length);

                patches.push_back(REFERENCE(image_patch));

                X_length+=stride;
            }
            Y_length+=stride;
        }

        window_length-=WINDOW_ENLARGING_CONST;
    }*/

    int size = patches.size(), count=0, hit_count=0;
    int number_of_samples = patches.size();

    double best_prediction = 0.0;

    //cout<<"---------SEGMENTATION DONE. Number of boxes: "<<size<<endl;
	
	if (ENABLE_PATCH_DISTRIBUTION) // Visualize sliding window captures. This is for testing purposes
	{
		for (cv::Mat P : patches)
		{
			namedWindow("Patches", WINDOW_AUTOSIZE);
			imshow("Patches", P);
			waitKey(0);
			destroyWindow("Patches");
		}
		//waitKey(0);
	}
	   
    while( (size/batch_limit) !=0 ) // Prevent memory overtake
    {
        std::vector<cv::Mat> portion(patches.begin()+count*batch_limit,patches.begin()+(count+1)*batch_limit);

        std::vector< std::vector<Prediction> > guess = batchPredict(portion,NUM_CLASSES,batch_limit);

        for(int i=0;i<guess.size();i++)
        {
          Mat im = portion[i];
          for(int j=0;j<guess[i].size();j++)
          {
            if(guess[i][j].first == TARGET_CLASS_STRING && guess[i][j].second>TARGET_PROBABILITY)
            {
				if (guess[i][j].second>best_prediction)
				{
					best_prediction = guess[i][j].second;
				}
				/*if (VISUALIZE_SWITCH)
				{
					std::cout << "Prediction: " << guess[i][j].second << std::endl;

					namedWindow("Pass", WINDOW_NORMAL);
					imshow("Pass", im);
					//waitKey(0);
					//destroyWindow("Pass"); //To see which part of the image caused detection
				}*/

              hit_count++;
              if((double)hit_count/*/(double)number_of_samples*/ > REQUIRED_PERCENTAGE)
              {
				  /*std::cout << "Best Prediction for Pizza: " << best_prediction * 100 << "%" << std::endl;
				  putText(resultMatrix, "Pizza Found! ", Point(1, massageHeight-4), FONT_HERSHEY_COMPLEX, 1.2, Scalar(0, 255, 0), 2);
				  vconcat(imageToDraw, resultMatrix, imageToDraw);
				  namedWindow("Result", WINDOW_NORMAL);
				  //imshow("Result", imageToDraw);
				  waitKey(WAIT_KEY_DELAY);
				  destroyWindow("Result"); //To see which part of the image caused detection
				  if (VISUALIZE_SWITCH)
				  destroyWindow("Pass"); //To see which part of the image caused detection*/
				return best_prediction;
              }
              break; // Go to next image
            }
            else if(guess[i][j].first == TARGET_CLASS_STRING && guess[i][j].second>0.0){
              if(guess[i][j].second>best_prediction)
              {
				  bestPrediction = im;
				  best_prediction = guess[i][j].second;
              }
            }
          }
        }

        count++;
        size-=batch_limit;
    }

    if(size%batch_limit!=0)
    {
        std::vector<cv::Mat> portion(patches.begin()+count*batch_limit,patches.end());

        std::vector< std::vector<Prediction> > guess = batchPredict(portion,NUM_CLASSES,(size%batch_limit));

        for(int i=0;i<guess.size();i++)
        {
          Mat im = portion[i];
          for(int j=0;j<guess[i].size();j++)
          {
            if(guess[i][j].first == TARGET_CLASS_STRING && guess[i][j].second>TARGET_PROBABILITY)
            {
				if (guess[i][j].second>best_prediction)
				{
					best_prediction = guess[i][j].second;
				}
				/*if (VISUALIZE_SWITCH)
				{
					std::cout << "Prediction: " << guess[i][j].second << std::endl;

					namedWindow("Pass", WINDOW_NORMAL);
					imshow("Pass", im);
					//waitKey(0);
					//destroyWindow("Pass"); //To see which part of the image caused detection
				}*/

              hit_count++;
              if((double)hit_count/*/(double)number_of_samples*/ > REQUIRED_PERCENTAGE)
              {
				  /*std::cout << "Best Prediction for Pizza: " << best_prediction * 100 << "%" << std::endl;
				  putText(resultMatrix, "Pizza Found!", Point(0, massageHeight - 4), FONT_HERSHEY_COMPLEX, 1.2, Scalar(0, 255, 0), 2);
				  vconcat(imageToDraw, resultMatrix, imageToDraw);
				  namedWindow("Result", WINDOW_NORMAL);
				  //imshow("Result", imageToDraw);
				  waitKey(WAIT_KEY_DELAY);
				  destroyWindow("Result"); //To see which part of the image caused detection
				  if (VISUALIZE_SWITCH)
				  destroyWindow("Pass"); //To see which part of the image caused detection*/
                  return best_prediction;
              }
              break;
            }
            else if(guess[i][j].first == TARGET_CLASS_STRING && guess[i][j].second>0.0){
              if(guess[i][j].second>best_prediction)
              {
				  bestPrediction = im;
				  best_prediction = guess[i][j].second;
              }
            }
          }
        }

    }
    
	/*std::cout << "Best Prediction for Pizza: " <<best_prediction * 100 << "%" << std::endl;
	putText(resultMatrix, "Negative! ", Point(2, massageHeight - 4), FONT_HERSHEY_COMPLEX, 1.2, Scalar(0, 0, 255), 2);
	vconcat(imageToDraw, resultMatrix, imageToDraw);
	namedWindow("Result1", WINDOW_NORMAL);
	//imshow("Result1", imageToDraw);
	if (VISUALIZE_SWITCH)
	{
		namedWindow("bestPrediction", WINDOW_NORMAL);
		imshow("bestPrediction", bestPrediction);
	}
	waitKey(WAIT_KEY_DELAY);
	destroyWindow("Result1"); //To see which part of the image caused detection
	if(VISUALIZE_SWITCH)
	destroyWindow("bestPrediction"); //To see which part of the image caused detection*/
    return best_prediction;

}

double caffeModel::operator<<(cv::Mat frame) // Analyze a sequence of frames and produce an overall guess
{
	bool result = guess(frame, BATCH_SIZE) > TARGET_PROBABILITY; 

	if (buffer.size() == BUFFER_SIZE)
	{
		successCounter -= buffer.front();
		buffer.pop();
	}

	buffer.push(result);
	successCounter += result;

	return (double)successCounter / (double)BUFFER_SIZE;
}

