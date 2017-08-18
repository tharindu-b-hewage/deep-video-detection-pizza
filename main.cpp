#include <iostream>
//#include <windows.h>
#include <ctime>
#include <thread>
#include <mutex>
#include "caffeModel.h"
#include <iostream>
#include <ctime>

#define FREE_TO_SET_FRAME				0
#define FRAME_LOCKED_FOR_USE			1
#define VIDEO_FINISHED					-1			// End of video file
#define EXECUTE_TIME_FLAG				true		// Show detection time. Warning this can cause stuttering video playback
#define IMAGE_CONFIDENCE_TRESHOLD		0.70		// Confidence ipper bound required for a single patch of pizza
#define QUEUE_LENGTH					10			// Length of the stored detection result
#define SEQUENCE_CONFIDENCE_THRESHOLD	0.90		// Confidence for stored detecction result queue

using namespace std;
using namespace cv;

mutex videoMutex, mainMutex;

int analyzeVideo(size_t* , cv::Mat* , double* , int );
std::string date_time(void);

string log_link = date_time() + "_pizza_logging.txt";
const char* l = std::string(date_time() + "_pizza_logging.txt").c_str();

std::ofstream logFile(l);

int main()
{
	//cout << "Link= " << l << endl;
	cout << "Please wait untill System loads.." << endl;

	caffeModel pizza_guy; //std::cout << "Here" << std::endl; std::cin.get();
//C:/Users/Tharindu Bandara/Desktop/Pack/x64/Release/caffeModelConfig.txt
    pizza_guy.initialize("C:/Users/Tharindu Bandara/Desktop/Pack/x64/Release/caffeModelConfig.txt"); // Generate caffe classifier


	string link = "C:/Users/Tharindu Bandara/Desktop/Research Project/Pizza Files/Test/Dialog Presentation";
	
	size_t status_flag = FREE_TO_SET_FRAME;

	cv::Mat globalFrame;
	double guessProbability;

	thread videoHandlerThread(analyzeVideo, &status_flag, &globalFrame, &guessProbability, 1); // Thread initiating 
 
	while (status_flag != VIDEO_FINISHED)
	{
		mainMutex.lock();
		//std::cout << "status flag : from main thread = " << status_flag << std::endl;
		mainMutex.unlock();

		if (status_flag == FRAME_LOCKED_FOR_USE)
		{
			mainMutex.lock();
			cv::Mat tempFrame = globalFrame;
			mainMutex.unlock();
			
			if (EXECUTE_TIME_FLAG)
			{
				mainMutex.lock();
				int start_s = clock();
				guessProbability = pizza_guy.guess(tempFrame, 50);
				int stop_s = clock();
				std::cout << "** Execution time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms"<< std::endl;
				mainMutex.unlock();
			}
			else
			{
				guessProbability = pizza_guy.guess(tempFrame, 50);
			}
			
			mainMutex.lock();
			//std::cout << "***************------------------------------  Guess from Main Thread:" << guessProbability << std::endl;
			status_flag = FREE_TO_SET_FRAME;
			mainMutex.unlock();
		}
	}
	if (videoHandlerThread.joinable())
		videoHandlerThread.join();
	

	/*while(1)
	{
		string number;
		cout<<"\nenter number: "; getline(cin,number); cout<<endl;
		string s = /*link+"/"+number+".jpg";
		//cout<<s<<endl;
		Mat image = imread(s,-1);
		if (image.data == NULL)
		{
			std::cout << "Mentioned image file cannot be read!" << endl;
			continue;
		}

		int start_s = clock();
		bool out = pizza_guy.guess(image, 200);
		int stop_s  = clock();

		if(out)
		{
			cout<<"\n  Result: Contains a Pizza \n"<<endl;
		}
		else
		{
			cout<<"\n  Result: Doesn't contain a  Pizza\n"<<endl;
		}

		cout << "---------Execution time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl; // << "--Sequence Confidence: " << (pizza_guy << image) << endl
	}*/

 return 0;
}


int analyzeVideo(size_t* aquireFlag, cv::Mat* globalFrame, double* confidence, int stride = 1)
{
	string link = "C:/Users/Tharindu Bandara/Desktop/Research Project/Pizza Files/Test/Dialog Presentation";
	videoMutex.lock();
	std::cout << "Enter video number with format: " << std::endl;
	string number;/* = "1.avi"; */ std::cin >> number;
	string videoURL = link + "/" + number;
	
	/*Logging for detection*/
	cout << date_time() + "_pizza_logging.txt";
	/*string log_link = date_time() + "_pizza_logging.txt";
	std::ofstream logFile( log_link.c_str() );*/
	/*Logging for detection*/
	if (!logFile.is_open())
		cout << "**********  Did Not Open !!!!!    ******************** \n";
	else
		cout << "**********  Successfull !!!!!    ********************\n";
	VideoCapture cap(videoURL); // open the video file for reading
	std::cout << videoURL << std::endl;
	videoMutex.unlock();
	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	std::queue<bool> sequenceResult; // Sequence analysis of pizza
	int sum = 0;

	//cap.set(CV_CAP_PROP_POS_MSEC, 300); //start the video at 300ms

	double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video

	double guessConfidence = *confidence;

	//cout << "Frame per seconds : " << fps << endl;

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

	cv::Mat frame;

	int count = 0;

	while (1)
	{

		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			videoMutex.lock();
			std::cout << "Cannot read the frame from video file" << endl;
			*aquireFlag = VIDEO_FINISHED;
			videoMutex.unlock();
			break;
		}

		//----- Modifications to the frame------
		//bool result = pizza.guess(frame, 100, false);
		//std::cout << result << std::endl;
		//std::cout << 
		if (count == stride)
		{
			//std::cout << "status flag video thread = " << aquireFlag << endl;
			if (*aquireFlag == FREE_TO_SET_FRAME)
			{
				videoMutex.lock();
				//std::cout << "Frame writing : from video thread" << endl;
				*globalFrame = frame;
				guessConfidence = *confidence;
				logFile << date_time() << " : " << guessConfidence * 100  ;
				*aquireFlag = FRAME_LOCKED_FOR_USE;
				videoMutex.unlock();
				//std::cout << "changed flag = " << *aquireFlag << endl;
			}
			
			count = 0;
		}
		else
			count++;

		//--------- Frame sequence analysis
		double sequencePizzaProb;
		if (sequenceResult.size() == QUEUE_LENGTH)
		{
			sum -= sequenceResult.front();
			sequenceResult.pop();
			sequenceResult.push(guessConfidence > IMAGE_CONFIDENCE_TRESHOLD);
			sum += sequenceResult.back();
			sequencePizzaProb = sum / (double)(QUEUE_LENGTH);
		}
		else
		{
			sequenceResult.push(guessConfidence > IMAGE_CONFIDENCE_TRESHOLD);
			sum += sequenceResult.back();
			sequencePizzaProb = sum / (double)(QUEUE_LENGTH);
		}//--------- Frame sequence analysis

		//--------------------------------------
		cv::Mat details(frame.rows * 0.2, frame.cols, CV_8UC3, Scalar(220, 220, 220));
		putText(details, to_string(guessConfidence*100)+"%", Point(0, 80), FONT_HERSHEY_COMPLEX, 1.2, Scalar(150, 155, 150), 2, true);
		
		if (sequencePizzaProb >= SEQUENCE_CONFIDENCE_THRESHOLD)
		{
			putText(details, "PIZZA!!!!", Point(frame.cols / 2, 80), FONT_HERSHEY_COMPLEX, 1.84, Scalar(0, 0, 255), 2.5, true);
			logFile << " : PIZZA!!";
		}
		videoMutex.lock();
		logFile << "\n";
		videoMutex.unlock();

		vconcat(frame, details, frame);
		imshow("MyVideo", frame); //show the frame in "MyVideo" window
		
		
		if (waitKey((size_t)(1000/fps)-10) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	

	videoMutex.lock();
	logFile.close();
	*aquireFlag = VIDEO_FINISHED;
	videoMutex.unlock();
	return 0;

}

std::string date_time(void)
{
	std::chrono::time_point<std::chrono::system_clock> current_date_time;
	current_date_time = std::chrono::system_clock::now();
	std::time_t end_time = std::chrono::system_clock::to_time_t(current_date_time);
	char* time = std::ctime(&end_time);
	for (int i=0;i<strlen(time);i++)
	{
		if (time[i] == ':') {
			time[i] = '_';
		}
		else if (time[i] == '\n') {
			time[i] = ' ';
		}
	}

	return std::string(time);
}