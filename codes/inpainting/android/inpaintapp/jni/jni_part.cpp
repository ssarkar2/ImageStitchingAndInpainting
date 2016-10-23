#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/photo/photo.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>

#include <string>

#include <android/log.h>

using namespace std;
using namespace cv;

extern "C" {

JNIEXPORT jint JNICALL Java_com_example_inpaintapp_MainActivity_inPaint(JNIEnv*, jobject, jstring scaledpath, jstring testpath, jstring maskpath, jint originalRectX, jint originalRectY, jint originalRectWidth, jint originalRectHeight);

JNIEXPORT jint JNICALL Java_com_example_inpaintapp_MainActivity_inPaint(JNIEnv *env, jobject, jstring scaledpath, jstring testpath, jstring maskpath, jint originalRectX, jint originalRectY, jint originalRectWidth, jint originalRectHeight)
{
	const char *nativescaledpath = env->GetStringUTFChars(scaledpath, JNI_FALSE);
	const char *nativetestpath = env->GetStringUTFChars(testpath, JNI_FALSE);
	const char *nativemaskpath = env->GetStringUTFChars(maskpath, JNI_FALSE);

	String scaledpath_str(nativescaledpath);
	Mat image;
	image = imread(scaledpath_str, CV_LOAD_IMAGE_COLOR);  //region of interest
	if(! image.data )                              // Check for invalid input
	{
	    cout <<  "Could not open or find the image" << std::endl ;
	    return -1;
	}
	String testpath_str(nativetestpath);
	Mat origimage = imread(testpath_str, CV_LOAD_IMAGE_COLOR);
	if(! origimage.data )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
	String maskpath_str(nativemaskpath);

	Mat blurred;
	GaussianBlur(image, blurred, Size( 5, 5 ), 0, 0 );

	Mat ycbcr;
	cvtColor(blurred,ycbcr,CV_RGB2YCrCb);
	Mat chan[3];
	split(ycbcr,chan);
	Mat y  = chan[0];

	Mat binary;
	cv::threshold(y, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	//foreground has less pixels
	unsigned char *input = (unsigned char*)(binary.data);
	int i,j,ch, zerocount = 0, onecount = 0;
	for(int i = 0;i < binary.cols;i++){
		for(int j = 0;j < binary.rows;j++){
			ch = input[binary.cols * j + i ];
			if (ch == 0)
				zerocount++;
			else
				onecount++;
		}
	}
	if (onecount > zerocount)
		bitwise_not ( binary, binary );

	Point topLeft; Point bottomRight;
	topLeft.x = originalRectX; topLeft.y = originalRectY;
	bottomRight.x = originalRectX + originalRectWidth;  bottomRight.y = originalRectY + originalRectHeight;
	//Mat mask(originalImageHeight, originalImageWidth, CV_8UC1, Scalar(0,0,0));
	Mat mask(origimage.rows, origimage.cols, CV_8UC1, Scalar(0,0,0));

	__android_log_print(ANDROID_LOG_VERBOSE, "inCCCC","topleftx = %d, toplefty = %d, bottomrightx = %d, bottomrighty = %d", topLeft.x, topLeft.y, bottomRight.x, bottomRight.y) ;
	__android_log_print(ANDROID_LOG_VERBOSE, "inCCCCx","origrow = %d, origcol = %d, maskrow = %d, maskcol = %d", origimage.rows, origimage.cols, mask.rows, mask.cols);
	__android_log_print(ANDROID_LOG_VERBOSE, "inCCCCxy", "%s", nativemaskpath);
	Mat mask_roi(mask, Rect(topLeft, bottomRight));
	binary.copyTo(mask_roi);



	Mat inpainted;

	inpaint(origimage, mask, inpainted, 10, cv::INPAINT_NS );
	imwrite( testpath_str, inpainted );

	imwrite( maskpath_str, mask );

	return 1;
}
}
