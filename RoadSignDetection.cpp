#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/* Global Variables */
Mat originalImage, sampleImage, groundTruthImage, redPixelImage, contourImage, threshOutputImage;
Vec3b red = { 0, 0, 255 };
Vec3b white = { 255, 255, 255 };
Vec3b black = { 0, 0, 0 };
Vec3b blue = { 232, 162, 0 };
const char* windowImage = "Result Image";

/* Function Headers */
Mat ReadImage(String);
Mat Histogram_BackProjection(Mat, Mat);
Mat AdaptiveThreshold(Mat);
Mat Contours(Mat);
Mat ChangeWhitePixelColour(Mat, Vec3b);
Mat kmeans_clustering(Mat&, int, int);
Mat ThresholdImage(Mat, int, int, int, int, int, int);
Mat ColourMaskOperations();
void CompareImages(Mat, Mat, Vec3b);

/* Main Function */
int main(int, char** argv)
{
	/* Read in the images */
	originalImage = ReadImage("C:/Users/Robert/Documents/Visual Studio 2015/Projects/OpenCVTest/OpenCVTest/Images/RoadSignsComposite1.JPG");
	sampleImage = ReadImage("C:/Users/Robert/Documents/Visual Studio 2015/Projects/OpenCVTest/OpenCVTest/Images/Sample.JPG");
	groundTruthImage = ReadImage("C:/Users/Robert/Documents/Visual Studio 2015/Projects/OpenCVTest/OpenCVTest/Images/RoadSignsCompositeGroundTruth.PNG");

	namedWindow(windowImage, WINDOW_AUTOSIZE);

	/* Perform k-means clustering on the original image */
	Mat clusteredImage = kmeans_clustering(originalImage, 45, 1);
	
	/* Convert the images from BGR to HLS */
	cvtColor(clusteredImage, clusteredImage, CV_BGR2HLS);
	cvtColor(sampleImage, sampleImage, CV_BGR2HLS);

	/* Threshold the image to identify the black pixels */
	threshOutputImage = ThresholdImage(clusteredImage, 0, 0, 0, 180, 83, 70);
	
	/* Convert the binary image to BGR so we can change the pixel colour later */
	cvtColor(threshOutputImage, threshOutputImage, CV_GRAY2BGR);

	/* Obtain a histogram from the sample image and back project the clustered original image */
	Mat backProjection = Histogram_BackProjection(clusteredImage, sampleImage);

	/* Change the white pixels in the back projected image to red */
	redPixelImage = ChangeWhitePixelColour(backProjection, red); 	/* Perform an adaptive threshold on the back projected image */	Mat adaptiveThreshimage = AdaptiveThreshold(backProjection);		/* Obtain the contour information of the adaptive thresholded image */	contourImage = Contours(adaptiveThreshimage);

	/* Change the colours of the image to the desired colours with various operations with different masks */
	Mat resultImage = ColourMaskOperations();

	/* Show the result image */
	imshow(windowImage, resultImage);

	/* Assess the classification performance */
	CompareImages(resultImage, groundTruthImage, red);
	CompareImages(resultImage, groundTruthImage, white);
	CompareImages(resultImage, groundTruthImage, black);

	waitKey(0);
	return 0;
}

/* Function to read in the images */
Mat ReadImage(String fileName) {
	
	Mat imageToRead;

	imageToRead = imread(fileName, CV_LOAD_IMAGE_COLOR);

	return imageToRead;
}

/*
* This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
* by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
*/
Mat kmeans_clustering(Mat& image, int k, int iterations)
{
	CV_Assert(image.type() == CV_8UC3);
	/* Populate an n*3 array of float for each of the n pixels in the image */
	Mat samples(image.rows*image.cols, image.channels(), CV_32F);
	float* sample = samples.ptr<float>(0);
	for (int row = 0; row<image.rows; row++)
		for (int col = 0; col<image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
				samples.at<float>(row*image.cols + col, channel) =
				(uchar)image.at<Vec3b>(row, col)[channel];
	/* Apply k-means clustering to cluster all the samples so that each sample
	   is given a label and each label corresponds to a cluster with a particular
	   centre. */
	Mat labels;
	Mat centres;
	kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1, 0.0001),
		iterations, KMEANS_PP_CENTERS, centres);
	/* Put the relevant cluster centre values into a result image */
	Mat& result_image = Mat(image.size(), image.type());
	for (int row = 0; row<image.rows; row++)
		for (int col = 0; col<image.cols; col++)
			for (int channel = 0; channel < image.channels(); channel++)
				result_image.at<Vec3b>(row, col)[channel] = (uchar)centres.at<float>(*(labels.ptr<int>(row*image.cols + col)), channel);
	return result_image;
}

/* Function that obtains a histogram of the sample image, backprojects the original image using the histogram and returns the binary image */
Mat Histogram_BackProjection(Mat image1, Mat image2) {
	
	Mat histogram;
	Mat hueImage1;
	Mat hueImage2;
	Mat backProjectionImage;
	const int* channel_numbers = {0};
	float channel_range[] = {0.0, 255.0};
	const float* channel_ranges = channel_range;
	int number_bins = 64;

	hueImage1.create(image1.size(), image1.depth());
	int channel1[] = { 0, 0 };
	mixChannels(&image1, 1, &hueImage1, 1, channel1, 1);

	hueImage2.create(image2.size(), image2.depth());
	int channel2[] = { 0, 0 };
	mixChannels(&image2, 1, &hueImage2, 1, channel2, 1);

	calcHist(&hueImage2, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	normalize(histogram, histogram, 0, 255, NORM_MINMAX, -1, Mat());

	calcBackProject(&hueImage1, 1, 0, histogram, backProjectionImage, &channel_ranges, 1, true);

	/* Dilates and erodes the image so that the holes can be filled in some of the signs that are difficult to classify */
	dilate(backProjectionImage, backProjectionImage, Mat(), Point(-1, -1), 6, 1, 1);
	erode(backProjectionImage, backProjectionImage, Mat(), Point(-1, -1), 5, 1, 1);
	
	return backProjectionImage;
}

/* Function to perform an adaptive threshold on an image */
Mat AdaptiveThreshold(Mat image) {
	
	Mat adaptiveThreshimage;

	adaptiveThreshold(image, adaptiveThreshimage, 255.0, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 5);

	return adaptiveThreshimage;
}

/* Function to locate, draw and fill the contours of an image */
Mat Contours(Mat image) {
	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(image, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	/* Draw contours */
	Mat drawing = Mat::zeros(image.size(), CV_8UC3);
	if (!contours.empty() && !hierarchy.empty()) {

		for (int i = 0; i < contours.size(); i++) {

			/*Look for hierarchy[i][3]!=-1 (hole boundaries) */
			if (hierarchy[i][3] != -1) {
				drawContours(drawing, contours, i, cvScalarAll(255), CV_FILLED);
			}
		}
	}
	return drawing;
}

/* Function to change all of the pixels classified as white to another colour */
Mat ChangeWhitePixelColour(Mat image, Vec3b colour) {

Mat colourImage; 
cvtColor(image, colourImage, CV_GRAY2BGR);

for (int i = 0; i < colourImage.rows; i++) {
	for (int j = 0; j < colourImage.cols; j++) {

		Vec3b newColour = colourImage.at<Vec3b>(i, j);

		if (newColour[0] >= 50 && newColour[1] >= 50 && newColour[2] >= 50) {
			newColour = colour;
			}
		colourImage.at<Vec3b>(i, j) = newColour;
		}
	}
	return colourImage;
}

/* Function to perform a binary threshold on an image */
Mat ThresholdImage(Mat Image, int lowH, int lowS, int lowL, int highH, int highS, int highL) {

	Mat threshImage;

	inRange(Image, Scalar(lowH, lowS, lowL), Scalar(highH, highS, highL), threshImage);

	bitwise_not(threshImage, threshImage);

	return threshImage;
}

/* Function to perform various operations to obtain the desired output */
Mat ColourMaskOperations() {
	
	Mat colourMaskImage = contourImage.clone();

	/* Obtains the white, blue and red pixels */
	for (int i = 0; i < colourMaskImage.rows; i++) {
		for (int j = 0; j < colourMaskImage.cols; j++) {

			Vec3b whiteMask = contourImage.at<Vec3b>(i, j);
			Vec3b redMask = redPixelImage.at<Vec3b>(i, j);
			Vec3b newColour = colourMaskImage.at<Vec3b>(i, j);

			if (whiteMask == white) {
				newColour = white;
			}
			else if (whiteMask == black) {
				newColour = blue;
			}
			if (redMask == red) {
				newColour = red;
			}
			colourMaskImage.at<Vec3b>(i, j) = newColour;
		}
	}

	Mat colourImage = colourMaskImage.clone();

	/* Obtains the black pixels */
	for (int i = 0; i < colourImage.rows; i++) {
		for (int j = 0; j < colourImage.cols; j++) {

			Vec3b mask = threshOutputImage.at<Vec3b>(i, j);
			Vec3b newColour = colourImage.at<Vec3b>(i, j);

			if (mask == black) {

				newColour = black;
			}
			colourImage.at<Vec3b>(i, j) = newColour;
		}
	}

	/* Adds another layer of blue and red pixels to remove the undesired black pixels */
	for (int i = 0; i < colourImage.rows; i++) {
		for (int j = 0; j < colourImage.cols; j++) {

			Vec3b redMask = redPixelImage.at<Vec3b>(i, j);
			Vec3b blueMask = colourMaskImage.at<Vec3b>(i, j);
			Vec3b newColour = colourImage.at<Vec3b>(i, j);

			if (redMask == red) {
				newColour = red;
			}
			if (blueMask == blue) {
				newColour = blue;
			}
			colourImage.at<Vec3b>(i, j) = newColour;
		}
	}
	return colourImage;
}

/* Assess the classification performance */
void CompareImages(Mat myImage, Mat groundTruth, Vec3b colour) {

	String compareColour;
	int falsePositives = 0;
	int falseNegatives = 0;
	int truePositives = 0;
	int trueNegatives = 0;

	/* Strings to use when outputting the results to the console window */
	if (colour == red) {
		compareColour = "RED";
	}
	else if (colour == white) {
		compareColour = "WHITE";
	}
	else {
		compareColour = "BLACK";
	}

	for (int i = 0; i < groundTruth.rows; i++) {
		for (int j = 0; j < groundTruth.cols; j++) {

			Vec3b result = myImage.at<Vec3b>(i, j);
			Vec3b gt = groundTruth.at<Vec3b>(i, j);

			if (result == colour && gt == colour) {
				truePositives++;
			}
			else if (result != colour && gt == colour) {
				falseNegatives++;
			}
			else if (result == colour && gt != colour) {
				falsePositives++;
			}
			else {
				trueNegatives++;
			}
		}
	}

	double precision = ((double)truePositives) / ((double)(truePositives + falsePositives));
	double recall = ((double)truePositives) / ((double)(truePositives + falseNegatives));
	double accuracy = ((double)(truePositives + trueNegatives)) / ((double)(truePositives + falsePositives + trueNegatives + falseNegatives));

	cout << "Precision of " << compareColour << " pixels: " << precision << endl;
	cout << "Recall of " << compareColour << " pixels: " << recall << endl;
	cout << "Accuracy of " << compareColour << " pixels: " << accuracy << endl << endl;
}