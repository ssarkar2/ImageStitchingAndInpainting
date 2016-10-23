#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <iostream>
#include <math.h> 

#define DISPLAY 0;
#define DISPLAY2 0;

using namespace cv;
using namespace std;

float confidenceScore(int *unfilledRegion, float* confidenceMtx, Point currPoint, int imageWidth, int imageHeight, int patchsize);
void plotNewFillFront(Mat image, vector<Point> unfilledContour, Point p);
void modifyFillFrontContour(vector<Point> *unfilledContour, int maxPriorityPatchNum, int* unfilledRegion, int patchsize, int imageWidth, int imageHeight);
void modifyFillFront(int *unfilledRegion, float* confidenceMtx, Point maxPriorityPatchCentre, Point exemplarLocation, int patchsize, int imageWidth,int *pixelsToBeFilled);
Point findExemplar(Mat patch, Point maxPriorityPatchCentre, Mat image, int* unfilledRegion, int skipSearch, int mode, float tolerance, float distThreshold);
void fillSobel(int* sobelX, int* sobelY, int size, float *maxSobelValue, int* sobelRadius, int* sobelSize);
float* findEdgeDirection(Mat imageY, Point p, int imageWidth, int imageHeight, int patchsize, int* unfilledRegion,  float* normalFillFront);
float* findNormalFillFront(int* unfilledRegion, Point p, int imageWidth, int imageHeight, int patchsize);
Point findPriorityPatch(int* unfilledRegion, Mat imageY, float *confidenceMtx, int patchsize, vector<Point> unfilledContour, int skip, int* maxPriorityPatchNum, int scoringMode);
Mat inpaint_exemplar(Mat image, Mat mask, vector<Point> polyCorner, int mode, Point inpaintregion, Mat origimg);
void createMaskAndInpaint(Mat image, double topLeftRow, double topLeftCol, double width, double height);
void onMouse(int evt, int x, int y, int flags, void* param);

void display(String s, Mat img);
Mat invertIfNeeded(Mat binary);
int roundPos(double n);
void inpaint_image(Mat image, Mat mask);
void drawRectOnImage(Mat image, double topLeftRow, double topLeftCol, double width, double height);
void drawRectOnImage1(Mat image, double topLeftRow, double topLeftCol, double width, double height);

//notes: after the polygon is clicked on, find a bounding box. 
//do canny in the bounding box... Canny will give the edges or isophotes
//maintain a matrix such that it tells us which parts of the canny edges are in unfilled region (therefore not available to us) and which parts are available.
//now when scanning the fill front, see the Canny map and find if an edge exists there and if its "valid" that is it does not lie in the unfilled part. (we can check in the direction of the normal of fillfront
//OR
//when scanning points in the fillfront, move a bit in the direction of the fillfront normal so that we are safely outside the unfilled region.
//now calc the gradient here and that will be high if theres an edge

int mode=100;

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: inpaint ImageToLoadAndDisplay mode" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	//resize(image, image, Size(700,700));

	cout << "size: "<< image.rows << " " << image.cols << endl;
	//int mode = *argv[2] - '0';

	createMaskAndInpaint(image, 0.75, 0.4, 0.55, 0.2);

    return 0;
}

void createMaskAndInpaint(Mat image, double topLeftRow, double topLeftCol, double width, double height)
{
	int imageWidth  = image.cols;
	int imageHeight = image.rows;
	Mat origimg = image.clone();
	Mat output = image;
	//while(1)
	{

	cv::Point2i pt(-1,-1);
	vector<Point> data;
	namedWindow( "original", WINDOW_AUTOSIZE );
	cv::setMouseCallback("original", onMouse, (void*)&data);
	imshow( "original", output );
	waitKey(0);
	Point inpaintregion;
	if (mode == 2 || mode == 3)
	{
		inpaintregion = data.back();
		data.pop_back();
	}

	if (mode == 1 || mode == 2)
	{
		Mat mask(imageHeight, imageWidth, CV_8UC1, Scalar(0,0,0));
		int sz = data.size();
		int numpoints[1]; numpoints[0] = data.size();
		const Point* ptsarray[1] = {&data[0]};
		fillPoly( mask, ptsarray, numpoints, 1, 255, 8 );
		display("poly", mask);

		fillPoly( image, ptsarray, numpoints, 1, Scalar(0,0,0), 8 );
		display("imagewithpoly", image);
		output = inpaint_exemplar(image, mask, data, mode, inpaintregion, origimg);
	}
	else
		output = inpaint_exemplar(image, Mat(), data, mode, inpaintregion, origimg);

	display("output", output);
	}

	//cout << contours.size() << endl;
   	cout << "bye" << endl;
}

Mat inpaint_exemplar(Mat image, Mat mask, vector<Point> polyCorner, int mode, Point inpaintregion, Mat origimg)
{
	int imageWidth  = image.cols;
	int imageHeight = image.rows;

	Mat imageYUV; cvtColor(image, imageYUV, CV_BGR2YCrCb);
	Mat channels[3];
	split( imageYUV, channels );
	Mat imageY = channels[0];

	float *confidenceMtx = (float*) malloc(sizeof(float) * imageWidth * imageHeight);
	int *unfilledRegion = (int*) malloc(sizeof(int) * imageWidth * imageHeight);  //its the mask of the unfilled region
	int *unfilledRegionOrig = (int*) malloc(sizeof(int) * imageWidth * imageHeight); //the source region
	int pixelsToBeFilled = 0;
	if(mode == 2 || mode == 3)//for mode 2 modify the mask and for mode 3 find the mask
	{
		Mat blurred;
		cout << inpaintregion <<endl;
		GaussianBlur(origimg, blurred, Size( 5, 5 ), 0, 0 );
		Mat ycbcr;
		cvtColor(blurred,ycbcr,CV_RGB2YCrCb);
		Mat chan[3];
		split(ycbcr,chan);
		Mat y  = chan[0];
		Mat binary(y.rows, y.cols, CV_8UC1, Scalar(0,0,0));
		//double thr = static_cast<double>(y.at<uchar>(inpaintregion.y, inpaintregion.x));
		double diff, maxdiff;
		double thr1 = 90.0, thr2 = 40;  //timestampimage
		//double thr1 = 30.0, thr2 = 10;  //barbara, blocky lena
		double refch0 = static_cast<double>(chan[0].at<uchar>(inpaintregion.y, inpaintregion.x));
		double refch1 = static_cast<double>(chan[1].at<uchar>(inpaintregion.y, inpaintregion.x));
		double refch2 = static_cast<double>(chan[2].at<uchar>(inpaintregion.y, inpaintregion.x));
		for (int r = 0; r < y.rows; r++)
		{
			for(int c = 0; c < y.cols; c++)
			{
				if (mode == 3 || mask.at<uchar>(r,c) == 255)
				{
					diff = pow((static_cast<double>(chan[0].at<uchar>(r, c))-refch0),2) + pow((static_cast<double>(chan[1].at<uchar>(r, c))-refch1),2) + pow((static_cast<double>(chan[2].at<uchar>(r, c))-refch2),2);
					maxdiff = max(max(abs(static_cast<double>(chan[0].at<uchar>(r, c))-refch0), abs(static_cast<double>(chan[1].at<uchar>(r, c))-refch1)), abs(static_cast<double>(chan[2].at<uchar>(r, c))-refch2));
					if (sqrt(diff) < thr1 && maxdiff < thr2)
						binary.at<uchar>(r,c) = 255;
					//cout << sqrt(diff) << " " << maxdiff <<endl;
				
				}
			}
		}
		display("mask", binary);
		dilate(binary, binary, Mat(), Point(-1, -1), 2, 1, 1);
		display("dilatemask", binary);
		mask = binary.clone();
		//display("mask1", binary);
		Mat maskedimg = origimg.clone();
		for (int r = 0; r < y.rows; r++)
		{
			for(int c = 0; c < y.cols; c++)
			{
				if (static_cast<int>(mask.at<uchar>(r,c)) > 0)
				{				
					maskedimg.at<Vec3b>(r,c) = Vec3b(0,0,0);
				}
			}
		}
		display("maskdimg", maskedimg);
		image = maskedimg.clone();
	}

	//initialization
	for (int i=0; i<imageWidth; i++)  //i is x axis
	{
		for (int j=0; j<imageHeight; j++) //j is y axis
		{
			if (mask.at<uchar>(j,i) == 255) //at accesses using row/col, not x/y
			{
				confidenceMtx[i + j*imageWidth] = 0;
				unfilledRegion[i + j*imageWidth] = 1;
				unfilledRegionOrig[i + j*imageWidth] = 1;
				pixelsToBeFilled++;
			}
			else
			{
				confidenceMtx[i + j*imageWidth] = 1;
				unfilledRegion[i + j*imageWidth] = 0;
				unfilledRegionOrig[i + j*imageWidth] = 0;
			}
		}
	}
	
	//find contour of initial mask
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	vector<Point> unfilledContour ;
	if (mode == 1)
	{
		findContours( mask.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );
		unfilledContour= contours[0];
	}
	else
	{
		Mat maskedge;
		Canny( mask, maskedge, 50, 150, 3);
		display("maskedge", maskedge);

		dilate(maskedge, maskedge, Mat(), Point(-1, -1), 2, 1, 1);
		erode(maskedge, maskedge, Mat(), Point(-1, -1), 2, 1, 1);
		display("maskedge1", maskedge);
		for (int r = 0; r < maskedge.rows; r++)
		{
			for (int c = 0; c < maskedge.cols; c++)
			{
				if (maskedge.at<uchar>(r,c) > 0)
					unfilledContour.push_back(Point(c,r));
			}
		}
	}

	
	/*for (int i = 0; i < unfilledContour.size(); i++)
	{
		circle(image, Point(unfilledContour[i]), 1, Scalar(255,0,0));
	}
	display("cont", image);*/

 	int size_ = unfilledContour.size();
	Mat imageOrig = image.clone();
#if DISPLAY
	for (auto it = begin (unfilledContour); it != end (unfilledContour); ++it)
	{
		line( image, *it, *it, CV_RGB( 255, 0, 0 ),  2, 8 );
		//unfilledContour.pop_back();
	}

	display ("cont", image);
#endif
	int patchsize = 9, patchradius = (patchsize-1)/2;
	int skipFillFront = 1; int skipSearch = patchradius;
	Point maxPriorityPatchCentre, exemplarLocation;
	Mat maxPriorityPatch;
	int maxPriorityPatchNum;

	while(pixelsToBeFilled > 0)
	{
		cout << "pixelsToBeFilled:" << pixelsToBeFilled <<endl;
		maxPriorityPatchCentre = findPriorityPatch(unfilledRegion, imageY, confidenceMtx, patchsize, unfilledContour, skipFillFront, &maxPriorityPatchNum, 2); //keep patchsize odd and >= 3
		maxPriorityPatch = image(Rect(Point(maxPriorityPatchCentre.x - patchradius , maxPriorityPatchCentre.y - patchradius), Point(maxPriorityPatchCentre.x + patchradius + 1 , maxPriorityPatchCentre.y + patchradius + 1)));  //note the +1 in bottom right corner

#if DISPLAY2
		//display("maxpatch", maxPriorityPatch);
		Mat B = image.clone();
		rectangle(B, Point(maxPriorityPatchCentre.x - (patchsize-1)/2, maxPriorityPatchCentre.y - (patchsize-1)/2), Point(maxPriorityPatchCentre.x + (patchsize-1)/2, maxPriorityPatchCentre.y + (patchsize-1)/2), Scalar(0,255,0), 1, 8, 0);
		display("maxpatchinimg", B);
#endif

		exemplarLocation = findExemplar(maxPriorityPatch, maxPriorityPatchCentre, image, unfilledRegion, skipSearch, 2, patchsize*patchsize*(1), patchsize*3);//patchsize*patchsize*(10 + 10 + 10));
#if DISPLAY2
		//display("bestmatch", image(Rect(exemplarLocation.x, exemplarLocation.y, patchsize, patchsize)));
		Mat C = image.clone();
		rectangle(C, exemplarLocation, Point(exemplarLocation.x + patchsize, exemplarLocation.y + patchsize), Scalar(255,0,255), 1, 8, 0);
		display("exemplarloc", C);
#endif 
		//cout << exemplarLocation << endl;

#if DISPLAY
		rectangle(image, exemplarLocation, Point(exemplarLocation.x + patchsize, exemplarLocation.y + patchsize), Scalar(255,0,0), 1, 8, 0);
		rectangle(image, Point(maxPriorityPatchCentre.x - (patchsize-1)/2, maxPriorityPatchCentre.y - (patchsize-1)/2), Point(maxPriorityPatchCentre.x + (patchsize-1)/2, maxPriorityPatchCentre.y + (patchsize-1)/2), Scalar(0,255,0), 1, 8, 0);
		display("rect", image);
#endif
		//copy patch
		image(Rect(exemplarLocation.x, exemplarLocation.y, patchsize, patchsize)).copyTo(image(Rect(Point(maxPriorityPatchCentre.x - patchradius , maxPriorityPatchCentre.y - patchradius), Point(maxPriorityPatchCentre.x + patchradius + 1 , maxPriorityPatchCentre.y + patchradius + 1))));

		modifyFillFront(unfilledRegion, confidenceMtx, maxPriorityPatchCentre, exemplarLocation, patchsize, imageWidth, &pixelsToBeFilled);
		Point p = unfilledContour[maxPriorityPatchNum];
		modifyFillFrontContour(&unfilledContour, maxPriorityPatchNum, unfilledRegion, patchsize, imageWidth, imageHeight);
#if DISPLAY
		plotNewFillFront(image, unfilledContour, p);
#endif
#if DISPLAY
		display ("image", image);
#endif
	}
	return image;
}

void plotNewFillFront(Mat image, vector<Point> unfilledContour, Point p)
{
	for (int i = 0; i < unfilledContour.size(); i++)
	{
		line(image, unfilledContour[i], unfilledContour[i],  CV_RGB( 0, 0, 255 ),  1, 8);
	}
	line(image, p, p,  CV_RGB( 0, 255, 0 ),  1, 8);

	display("newfront", image);
}

void modifyFillFrontContour(vector<Point> *unfilledContour, int maxPriorityPatchNum, int* unfilledRegion, int patchsize, int imageWidth, int imageHeight)
{
	Point maxPriorityPatchCentre = (*unfilledContour)[maxPriorityPatchNum];
	int patchRadius = (patchsize-1)/2;

	//iterate along boundaries of the newly inserted patch and see if those points are in the interior or the border of the known region.
	//check top and bottom boundary
	for (int i = -patchRadius-1; i <= patchRadius+1; i++) //i is the x axis
	{
		//check the points above the top boundary
		if (maxPriorityPatchCentre.y - patchRadius - 1 >= 0 && maxPriorityPatchCentre.x - patchRadius - 1 >= 0 && maxPriorityPatchCentre.x + patchRadius + 1 < imageWidth)
		{
			if ((*(unfilledRegion + (maxPriorityPatchCentre.y - patchRadius - 1)*imageWidth + maxPriorityPatchCentre.x + i) == 1))
			{
				(*unfilledContour).push_back(Point(maxPriorityPatchCentre.x + i, maxPriorityPatchCentre.y - patchRadius));
			}
		}

		//check the points below the bottom boundary
		if (maxPriorityPatchCentre.y + patchRadius + 1 < imageHeight && maxPriorityPatchCentre.x - patchRadius - 1 >= 0 && maxPriorityPatchCentre.x + patchRadius + 1 < imageWidth)
		{
			if ((*(unfilledRegion + (maxPriorityPatchCentre.y + patchsize + 1)*imageWidth + maxPriorityPatchCentre.x + i) == 1))
			{
				(*unfilledContour).push_back(Point(maxPriorityPatchCentre.x + i, maxPriorityPatchCentre.y + patchRadius));
			}
		}
	}

	//check left and right boundaries
	for (int i = -patchRadius-1; i <= patchRadius+1; i++) //i is the x axis
	{
		//check the points on the left boundary
		if (maxPriorityPatchCentre.x - patchRadius - 1 >= 0 && maxPriorityPatchCentre.y - patchRadius - 1 >= 0 && maxPriorityPatchCentre.y + patchRadius + 1 < imageHeight)
		{
			if ((*(unfilledRegion + (maxPriorityPatchCentre.y  + i)*imageWidth + maxPriorityPatchCentre.x - patchRadius - 1) == 1))
			{
				(*unfilledContour).push_back(Point(maxPriorityPatchCentre.x - patchRadius, maxPriorityPatchCentre.y + i));
			}
		}

		//check the points on the right boundary
		if (maxPriorityPatchCentre.x + patchRadius + 1 < imageWidth && maxPriorityPatchCentre.y - patchRadius - 1 >= 0 && maxPriorityPatchCentre.y + patchRadius + 1 < imageHeight)
		{
			if ((*(unfilledRegion + (maxPriorityPatchCentre.y  + i)*imageWidth + maxPriorityPatchCentre.x + patchRadius + 1) == 1))
			{
				(*unfilledContour).push_back(Point(maxPriorityPatchCentre.x + patchRadius, maxPriorityPatchCentre.y + i));
			}
		}
	}
	
	//delete
	Point centre = (*unfilledContour)[maxPriorityPatchNum];
	int iter = 0;
	for (int i = 0; i < (*unfilledContour).size(); )
	{
		if ((*unfilledContour)[i].x > centre.x - patchRadius && (*unfilledContour)[i].x < centre.x + patchRadius && (*unfilledContour)[i].y > centre.y - patchRadius && (*unfilledContour)[i].y < centre.y + patchRadius)
		{
			(*unfilledContour).erase((*unfilledContour).begin() + i);	  ///check if deletion is hapenning
		}
		else
			i++;
	}


}

void modifyFillFront(int *unfilledRegion, float* confidenceMtx, Point maxPriorityPatchCentre, Point exemplarLocation, int patchsize, int imageWidth,int *pixelsToBeFilled)
{
	int patchRadius = (patchsize-1)/2;
	for (int i = -patchRadius ; i <= patchRadius; i++)  //i is x axis
	{
		for (int j = -patchRadius ; j <= patchRadius; j++) //j is y axis
		{
			if (*(unfilledRegion + (maxPriorityPatchCentre.y + j)*imageWidth + maxPriorityPatchCentre.x + i) == 1) 
			{
				*pixelsToBeFilled = *pixelsToBeFilled - 1;
				*(confidenceMtx + (maxPriorityPatchCentre.y + j)*imageWidth + maxPriorityPatchCentre.x + i) = *(confidenceMtx + (exemplarLocation.y + patchRadius + j)*imageWidth + exemplarLocation.x + patchRadius + i);
			}
			*(unfilledRegion + (maxPriorityPatchCentre.y + j)*imageWidth + maxPriorityPatchCentre.x + i) = 0;
		}
	}
}

Point findExemplar(Mat patch, Point maxPriorityPatchCentre, Mat image, int* unfilledRegion, int skipSearch, int mode, float tolerance, float distThreshold) //mode 1 uses YUV for comparision, mode 2 uses RGB
{
	int patchsize = patch.rows;
	int patchRadius = (patchsize-1)/2;
	int imageWidth  = image.cols;
	int imageHeight = image.rows;
	int i = 0, j = 0;
	double minDiff = 10000000000;
	Point topLeft, bottomRight;
	Scalar s;
	Mat diff;
	Mat patchYCbCr, imageYCbCr;
	float channel0Weight = 0, channel1Weight = 0, channel2Weight = 0;
	float diffValue;
	Point minPoint;
	int origSkip = skipSearch;
	int fineCombMode = 0, fineCombModeCount = 0;
	int breaknow = 0;
	int lowThreshold;
	Mat imagePatchedges;
	int edgecount, prevedgecount = 0;
	float distCandidateToFill = 10000000000, prevdistCandidateToFill = 10000000000;
	float oldDiff = 0;

	int lastCombedAti = 0, lastCombedAtj = 0, window_i = origSkip, window_j = origSkip;

	Mat testImg, testPatch, imagePatch;

	cvtColor(image, imageYCbCr, CV_RGB2YCrCb);
	
	vector<Mat> imageYUVchannels(3);
	split(imageYCbCr, imageYUVchannels);
	Mat imageY = imageYUVchannels[0];

	switch(mode)
	{
	case 1:
		cvtColor(patch, patchYCbCr, CV_RGB2YCrCb);
		testImg = imageYCbCr; testPatch = patchYCbCr;
		channel0Weight = 2.0/4, channel1Weight = 1.0/4, channel2Weight = 1.0/4;  //Y is given more weight
		break;
	case 2:
		testImg = image; testPatch = patch;
		channel0Weight = 1.0/3, channel1Weight = 1.0/3, channel2Weight = 1.0/3;
		break;
	default:
		break;
	}

	vector<Mat> testchannels(3);
	split(testPatch, testchannels);

	int breakFlag = 0; //goto next loop if exemplar is not fully inside known region
	while(i < imageHeight-patchsize) //i is y axis  //i and j are coordinates of topleft of exemplar candidate
	{
		while (j < imageWidth-patchsize)//j is x axis
		{
			breakFlag = 0;
			topLeft = Point(j,i);  //CHECK LATER remove these
			bottomRight = Point(j+patchsize , i+patchsize );
			imagePatch = testImg(Rect(j, i, patchsize, patchsize));
			//check that exempler lies in known zone
			//check that all the border points of the patch lie inside known zone
			//check top row and bottom row
			for (int m = 0; m < imagePatch.rows; m++)
			{
				if (*(unfilledRegion + (i)*imageWidth + (j+m)) == 1 || *(unfilledRegion + (i+patchsize)*imageWidth + (j+m)) == 1)
				{
					breakFlag = 1;
					break;
				}
			}
			if (breakFlag == 1) //continue the inner while loop
			{
				j+=skipSearch;
				continue;
			}
			//check left and right borders
			for (int m = 0; m < imagePatch.rows; m++)
			{
				if (*(unfilledRegion + (i+m)*imageWidth + (j) ) == 1 || *(unfilledRegion + (i+m)*imageWidth + (j+patchsize)) == 1)
				{
					breakFlag = 1;
					break;
				}
			}
			if (breakFlag == 1)//continue the inner while loop
			{
				j+=skipSearch;
				continue;
			}

			vector<Mat> imgchannels(3);
			split(imagePatch, imgchannels);
			
			float ch0 = 0, ch1 = 0, ch2 = 0, dch0 = 0, dch1 = 0, dch2 = 0;
			for (int p = 0; p < patchsize; p++) //y axis
			{
				for (int q = 0; q < patchsize; q++) //x axis
				{
					if (*(unfilledRegion + (maxPriorityPatchCentre.y - patchRadius + p)*imageWidth + maxPriorityPatchCentre.x - patchRadius + q) == 0)
					{
						dch0 = abs(float(testchannels[0].at<uchar>(p, q)) - float(imgchannels[0].at<uchar>(p, q)));
						dch1 = abs(float(testchannels[1].at<uchar>(p, q)) - float(imgchannels[1].at<uchar>(p, q)));
						dch2 = abs(float(testchannels[2].at<uchar>(p, q)) - float(imgchannels[2].at<uchar>(p, q)));;
						ch0 += dch0;
						ch1 += dch1;
						ch2 += dch2;
					}
				}
			}

			diffValue = ch0 * channel0Weight + ch1 * channel1Weight + ch2 * channel2Weight;

			if(diffValue <= minDiff)
			{
				//HACK: check later.... check that edges hitting the mask boundary are propagated

				//5... keep patches that are closer
				distCandidateToFill = sqrt(float((maxPriorityPatchCentre.x - patchsize - j)*(maxPriorityPatchCentre.x - patchsize - j)) + float((maxPriorityPatchCentre.y - patchsize - i)*(maxPriorityPatchCentre.y - patchsize - i)));

				if (diffValue < minDiff || (distCandidateToFill < prevdistCandidateToFill && diffValue == minDiff))
				{
					if (diffValue == minDiff)
					{
						prevdistCandidateToFill = distCandidateToFill;
					}

					oldDiff = minDiff;
					minDiff = diffValue;
					minPoint.x = j; minPoint.y = i;

					if (ch0 < tolerance && ch1 < tolerance && ch2 < tolerance && distCandidateToFill < distThreshold) //Chekck later: remove this if needed
					{
						breaknow = 1;
					}

					//hack:  if we just found a minimum, then retrace steps and go over that region with a finer comb   //CHECK LATER

					if (breaknow ==1)
						break;

					if (diffValue < oldDiff)
					{
						if (fineCombMode == 0 && j - (origSkip+1) >= 0 && i - (origSkip+1) >= 0)
						{
							
							window_i = min((origSkip+1), (i - lastCombedAti + 1));
							window_j = min((origSkip+1), (j - lastCombedAtj + 1));
							lastCombedAtj = j; lastCombedAti = i;
							j = j - window_i;
							i = i - window_j;
							skipSearch = 1;
							fineCombMode = 1;
							continue;
						}
					}
				}
			}

			if (fineCombMode == 1)
			{
				fineCombModeCount++;
				if(fineCombModeCount > (window_i)*(window_j))
				{
					skipSearch = origSkip;
					fineCombMode = 0;
					fineCombModeCount = 0;
				}
			}
			j+=skipSearch;
		}
		if (breaknow ==1) break;
		i+=skipSearch;
		j = 0;
		lastCombedAtj = 0;
	}
	return minPoint;
}

//unfilledRegion is a binary mask, patch size is dimension of patches we will use for exemplar comparision, unfilledContour is a vector of points on the fill front over whihc we traverse ..skip is how many points in unfilledContour we skip... skip = n means 1 in n are processed
Point findPriorityPatch(int* unfilledRegion, Mat imageY, float *confidenceMtx, int patchsize, vector<Point> unfilledContour, int skip, int* maxPriorityPatchNum, int scoringMode)  //scoringMode = 1: only D is used, 2 only C is used, 3 both are used
{
	int imageWidth  = imageY.cols;
	int imageHeight = imageY.rows;
	float D = 1, maxD = -1000000; 
	float C = 1, maxC = -1000000; 
	int P = 1,  maxP = -1000000; 
	Point maxPoint;

	cout << "unfilled contour size " <<unfilledContour.size() <<endl;
	Mat currpatchY;
	Point topLeft, bottomRight;
	float* normalFillFront; //normal to the fillfront at a particular point.  corresponds to np 
	float* edgeDirection; //corresponding to delta(Ip)
	Point currPoint;
	int sum = 0;

	Point it;
	for (int vectCount = 0; vectCount<unfilledContour.size(); vectCount+=skip)
	{
		it = unfilledContour[vectCount];

		if (it.x - (patchsize-1)/2 > 0 && it.y - (patchsize-1)/2 > 0 && it.x + (patchsize-1)/2 < imageWidth && it.y + (patchsize-1)/2 < imageHeight)  //bounds check
		{
			topLeft.x = it.x - (patchsize-1)/2; topLeft.y = it.y - (patchsize-1)/2;
			bottomRight.x = it.x + (patchsize-1)/2 + 1;  bottomRight.y = it.y + (patchsize-1)/2 + 1;//note the +1 
		}
		else
			continue;

		currpatchY = imageY(Rect(topLeft, bottomRight));
		currPoint.x = it.x; currPoint.y = it.y;


		if (scoringMode == 1 || scoringMode == 3)
		{
			normalFillFront = findNormalFillFront(unfilledRegion, currPoint, imageWidth, imageHeight, patchsize); //np
			edgeDirection = findEdgeDirection(imageY, currPoint, imageWidth, imageHeight, patchsize, unfilledRegion, normalFillFront); //del Ip
			D = abs(normalFillFront[0] * edgeDirection[0] + normalFillFront[1] * edgeDirection[1])/255;
		}
		if (scoringMode == 2 || scoringMode == 3)
		{
			C = confidenceScore(unfilledRegion, confidenceMtx, currPoint, imageWidth, imageHeight, patchsize);
		}
		P = D*C;  //priority

		if (P > maxP)
		{
			maxP = P;
			maxPoint.x = it.x ; maxPoint.y = it.y ;
			*maxPriorityPatchNum = vectCount;
		}
	}
	return maxPoint;
}

float confidenceScore(int *unfilledRegion, float* confidenceMtx, Point currPoint, int imageWidth, int imageHeight, int patchsize)
{
	int patchradius = (patchsize-1)/2;
	int startx = -patchradius, stopx = patchradius, starty = -patchradius, stopy = patchradius;
	float conf = 0;
	if (currPoint.x + startx < 0) startx = -currPoint.x ;  //so that currPoint.x + startx starts at 0
	if (currPoint.y + starty < 0) starty = -currPoint.y ;
	if (currPoint.x + stopx > imageWidth) stopx = imageWidth - currPoint.x;
	if (currPoint.y + stopy > imageHeight) stopy = imageHeight - currPoint.y;

	for (int i = startx; i <= stopx; i++) //x
	{
		for (int j = starty; j <= stopy; j++) //y
		{
			if (*(unfilledRegion + (currPoint.y + starty + j) * imageWidth + currPoint.x + startx + i) == 0)
			{
				conf += *(confidenceMtx + (currPoint.y + starty + j) * imageWidth + currPoint.x + startx + i);
			}
		}
	}

	return conf/((stopx - startx + 1)*(stopy - starty + 1));
}

float* findEdgeDirection(Mat imageY, Point p, int imageWidth, int imageHeight, int patchsize, int* unfilledRegion, float* normalFillFront)
{
	int* sobelX, *sobelY; int sobelRadius = 0, sobelSize = 0;
	int patchRadius = (patchsize-1)/2;
	float maxSobelValue = 0;
	//define Sobel filters for x and y directions
	if (patchsize == 3)
	{
		sobelX = (int*)malloc(9*sizeof(int)); sobelY = (int*)malloc(9*sizeof(int));
		fillSobel(sobelX, sobelY, 3, &maxSobelValue, &sobelRadius, &sobelSize);
	}
	else  //patchsize > 3...use a 5x5 sobel filter
	{
		sobelX = (int*)malloc(25*sizeof(int)); sobelY = (int*)malloc(25*sizeof(int));
		fillSobel(sobelX, sobelY, 5, &maxSobelValue, &sobelRadius, &sobelSize);
	}

	Point patchTopLeft, patchBottomRight;
	patchTopLeft.x = p.x - (sobelSize-1)/2; patchTopLeft.y = p.y - (sobelSize-1)/2;
	patchBottomRight.x = p.x + (sobelSize-1)/2 + 1; patchBottomRight.y = p.y + (sobelSize-1)/2 + 1;//note the +1

	Mat patch;// = imageY(Rect(patchTopLeft, patchBottomRight));
	float fillFrontGradAngle = (180/3.14159265358979323846) * atan2(normalFillFront[1], normalFillFront[0]);
	int movex = 0, movey = 0; int movedistx = 5;int movedisty = 5;
	if (fillFrontGradAngle > -22.5 && fillFrontGradAngle <= 22.5)
	{
		if (p.x - movedistx - sobelRadius> 0)
			movex = -movedistx; 
		movey = 0;
	}
	if (fillFrontGradAngle > 22.5 && fillFrontGradAngle <= 67.5)
	{
		if (p.x - movedistx - sobelRadius > 0)
			movex = -movedistx; 
		if (p.y - movedisty - sobelRadius > 0)
			movey = -movedisty;
	}
	if (fillFrontGradAngle > 67.5 && fillFrontGradAngle <= 112.5)
	{
		movex = 0; 
		if (p.y - movedisty - sobelRadius> 0)
			movey = -movedisty;
	}
	if (fillFrontGradAngle > 112.5 && fillFrontGradAngle <= 157.5)
	{
		if (p.x + movedistx + sobelRadius < imageWidth)
			movex = movedistx; 
		if (p.y - movedisty - sobelRadius> 0)
			movey = -movedisty;
	}
	if (fillFrontGradAngle <= -22.5 && fillFrontGradAngle > -67.5)
	{
		if (p.x - movedistx - sobelRadius> 0)
			movex = -movedistx; 
		if (p.y + movedisty + sobelRadius < imageHeight)
			movey = movedisty;
	}
	if (fillFrontGradAngle <= -67.5 && fillFrontGradAngle > -112.5)
	{
		movex = 0; 
		if (p.y + movedisty + sobelRadius < imageHeight)
			movey = movedisty;
	}
	if (fillFrontGradAngle <= -112.5 && fillFrontGradAngle > -157.5)
	{
		if (p.x + movedistx + sobelRadius < imageWidth)
			movex = movedistx; 
		if (p.y + movedisty + sobelRadius < imageHeight)
			movey = movedisty;
	}
	if (fillFrontGradAngle <= -157.5 || fillFrontGradAngle > 157.5)
	{
		if (p.x + movedistx + sobelRadius < imageWidth)
			movex = movedistx; 
		movey = 0;
	}

	patch = imageY(Rect(Point(p.x+movex - sobelRadius, p.y+movey - sobelRadius),  Point(p.x+movex + sobelRadius + 1, p.y+movey + sobelRadius + 1) ));

	int* maskpatch = (int*) malloc(sizeof(int) * sobelSize * sobelSize);
	int startCol = p.x+movex - sobelRadius; int startRow = p.y+movey - sobelRadius;
	for (int i = 0; i < sobelSize; i++) //i is x
	{
		//memcpy(maskpatch + sobelSize*i, (imageY.data + (startRow+i) * imageWidth + startCol), sobelSize); //copying patch row by row
		for (int j = 0; j < sobelSize; j++) //j is y
			*(maskpatch +i +  sobelSize*j) = *(unfilledRegion + (startRow+j)*imageWidth + startCol+i);// imageY.at<uchar>(startRow+j, startCol+i);
	}


	//important for demo
	/*Mat imgg = imageY;
	//rectangle(imgg, Point(p.x+movex - sobelRadius, p.x+movey - sobelRadius),  Point(p.x+movex + sobelRadius + 1, p.x+movey + sobelRadius + 1), Scalar( 0, 255, 255 ), 1,8);
	line(imgg, p, p, 100);
	rectangle(imgg, Point(p.x+movex - sobelRadius, p.y+movey - sobelRadius),  Point(p.x+movex + sobelRadius + 1, p.y+movey + sobelRadius + 1), 200);
	display("imgg", imgg);*/
	      
	float xsum = 0, ysum = 0;
	for (int i = 0; i < sobelSize; i++)  //i is x
	{
		for (int j = 0; j < sobelSize; j++) //j is y
		{
			if (*(maskpatch + i + j*sobelSize) == 0) //if its part of filled region, use the point to calculate gradient
			{
				xsum = xsum +  sobelX[i * sobelSize + j] * float(patch.at<uchar>(j,i));
				ysum = ysum +  sobelY[i * sobelSize + j] * float(patch.at<uchar>(j,i));
			}
			else
			{
				//cout << " notgood "  ;
			}
		}
	}
	float *edgeDirection = (float*)malloc(sizeof(float)*2);
	/*if (xsum == 0 && ysum == 0)
	{
		edgeDirection[0] = 0; edgeDirection[1] = 0;
	}
	else
	{
		edgeDirection[0] = xsum/(sqrt(xsum*xsum + ysum*ysum)); edgeDirection[1] = ysum/(sqrt(xsum*xsum + ysum*ysum));
	}*/

	edgeDirection[0] = xsum; edgeDirection[1] = ysum;
	return edgeDirection;
}

//takes the unfilled region mask and finds the normal for the boundary separating filled and unfilled region at point p
float* findNormalFillFront(int* unfilledRegion, Point p, int imageWidth, int imageHeight, int patchsize)
{
	int* sobelX, *sobelY; int sobelRadius = 0, sobelSize = 0;
	int patchRadius = (patchsize-1)/2;
	float maxSobelValue = 0;
	//define Sobel filters for x and y directions
	if (patchsize == 3)
	{
		sobelX = (int*)malloc(9*sizeof(int)); sobelY = (int*)malloc(9*sizeof(int));
		fillSobel(sobelX, sobelY, 3, &maxSobelValue, &sobelRadius, &sobelSize);
	}
	else  //patchsize > 3...use a 5x5 sobel filter
	{
		sobelX = (int*)malloc(25*sizeof(int)); sobelY = (int*)malloc(25*sizeof(int));
		fillSobel(sobelX, sobelY, 5, &maxSobelValue, &sobelRadius, &sobelSize);
	}

	int* maskpatch = (int*) malloc(sizeof(int) * sobelSize * sobelSize);
	int startCol = p.x - sobelRadius; int startRow = p.y - sobelRadius;
	for (int i = 0; i < sobelSize; i++)
	{
		for (int j = 0; j < sobelSize; j++)
		{
			*(maskpatch + sobelSize*i + j) = *(unfilledRegion + (startRow+i) * imageWidth + startCol + j);
		}
	}
	
	//apply x gradient and y gradient operator at centre point of patch
	float xsum = 0, ysum = 0;
	for (int i = 0; i < sobelSize; i++)
	{
		for (int j = 0; j < sobelSize; j++)
		{
			xsum = xsum + 255*maskpatch[sobelSize*i +  j] * sobelX[i * sobelSize + j];
			ysum = ysum + 255*maskpatch[sobelSize*i +  j] * sobelY[i * sobelSize + j];
		}
	}
	
	float *normalFillFront = (float*)malloc(sizeof(float)*2);
	if (xsum == 0 && ysum == 0)
	{
		normalFillFront[0] = 0; normalFillFront[1] = 0;
	}
	else
	{
		normalFillFront[0] = xsum/(sqrt(xsum*xsum + ysum*ysum)); normalFillFront[1] = ysum/(sqrt(xsum*xsum + ysum*ysum));
	}
	return normalFillFront;
}

void fillSobel(int* sobelX, int* sobelY, int size, float *maxSobelValue, int* sobelRadius, int* sobelSize)
{
	if (size == 3)
	{
		*sobelRadius = 1; *sobelSize = 3;
		sobelX[0] = -1; sobelX[1] = 0; sobelX[2] = 1;
		sobelX[3] = -2; sobelX[4] = 0; sobelX[5] = 2;
		sobelX[6] = -1; sobelX[7] = 0; sobelX[8] = 1;

		sobelY[0] = -1; sobelY[1] = -2; sobelY[2] = -1;
		sobelY[3] = 0; sobelY[4] = 0; sobelY[5] = 0;
		sobelY[6] = 1; sobelY[7] = 2; sobelY[8] = 1;

		*maxSobelValue = 4;
	}
	else  //patchsize > 3...use a 5x5 sobel filter
	{
		*sobelRadius = 2; *sobelSize = 5;
		sobelX[0] = -1; sobelX[1] = -2; sobelX[2] = 0; sobelX[3] = 2; sobelX[4] = 1;
		sobelX[5] = -4; sobelX[6] = -8; sobelX[7] = 0; sobelX[8] = 8; sobelX[9] = 4;
		sobelX[10] = -6; sobelX[11] = -12; sobelX[12] = 0; sobelX[13] = 12; sobelX[14] = 6;
		sobelX[15] = -4; sobelX[16] = -8; sobelX[17] = 0; sobelX[18] = 8; sobelX[19] = 4;
		sobelX[20] = -1; sobelX[21] = -2; sobelX[22] = 0; sobelX[23] = 2; sobelX[24] = 1;

		sobelY[0] = -1; sobelY[1] = -4; sobelY[2] = -6; sobelY[3] = -4; sobelY[4] = -1;
		sobelY[5] = -2; sobelY[6] = -8; sobelY[7] = -12; sobelY[8] = -8; sobelY[9] = -2;
		sobelY[10] = 0; sobelY[11] = 0; sobelY[12] = 0; sobelY[13] = 0; sobelY[14] = 0;
		sobelY[15] = 2; sobelY[16] = 8; sobelY[17] = 12; sobelY[18] = 8; sobelY[19] = 2;
		sobelY[20] = 1; sobelY[21] = 4; sobelY[22] = 6; sobelY[23] = 4; sobelY[24] = 1;

		*maxSobelValue = 48;
	}
}

void onMouse(int evt, int x, int y, int flags, void* param) {
	static int down = 0;
	static int count =0;
	if (evt == EVENT_LBUTTONDOWN) down = 1;
	if (evt == EVENT_LBUTTONUP) down = 0;

	if ( flags == (EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON) )
	{
		vector<Point>* ptPtr = (vector<Point>*)param;
        ptPtr->push_back(Point(x,y));
		count++;
		cout << count << endl;
		cout << x << " " << y << endl;
		mode = 3;
		return;
	}
    if(evt == EVENT_LBUTTONDOWN ) {
        vector<Point>* ptPtr = (vector<Point>*)param;
        ptPtr->push_back(Point(x,y));
		count++;
		cout << count << endl;
		mode = 1;
		return;
    }
	if(evt == EVENT_RBUTTONDOWN ) 
	{
		vector<Point>* ptPtr = (vector<Point>*)param;
        ptPtr->push_back(Point(x,y));
		count++;
		cout << count << endl;
		mode = 2;
		return;
	}
}
void drawRectOnImage1(Mat image, double topLeftRow, double topLeftCol, double width, double height)
{
	int imageWidth  = image.cols;
	int imageHeight = image.rows;

	Point topLeft; topLeft.x = roundPos(topLeftCol*imageWidth); topLeft.y = roundPos(topLeftRow*imageHeight);
	Point bottomRight; bottomRight.x = roundPos((topLeftCol + width)*imageWidth) ; bottomRight.y = roundPos((topLeftRow + height)*imageHeight) ; 

	display("orig", image);

	Mat miniMat = image(Rect(topLeft, bottomRight));

	Mat blurred = miniMat;
	GaussianBlur(miniMat, blurred, Size( 5, 5 ), 0, 0 );

	Mat chan[3];
	split(blurred,chan);

	Mat binaryR;
	cv::threshold(chan[0], binaryR, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	display("binaryR", binaryR);
	binaryR = invertIfNeeded(binaryR);
	display("binaryR1", binaryR);

	Mat binaryG;
	cv::threshold(chan[1], binaryG, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	display("binaryG", binaryG); 
	binaryG = invertIfNeeded(binaryG);
	display("binaryG1", binaryG);

	Mat binaryB;
	cv::threshold(chan[2], binaryB, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	display("binaryB", binaryB); 
	binaryB = invertIfNeeded(binaryB);
	display("binaryB1", binaryB);

	//combineMask(binaryR, binaryG, binaryB);
}

Mat invertIfNeeded(Mat binary)
{
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
	return binary;
}

void drawRectOnImage(Mat image, double topLeftRow, double topLeftCol, double width, double height)
{
	int imageWidth  = image.cols;
	int imageHeight = image.rows;

	Point topLeft; topLeft.x = roundPos(topLeftCol*imageWidth); topLeft.y = roundPos(topLeftRow*imageHeight);
	Point bottomRight; bottomRight.x = roundPos((topLeftCol + width)*imageWidth) ; bottomRight.y = roundPos((topLeftRow + height)*imageHeight) ; //note the +1

	//int thickness=1; int lineType=8; int shift=0;
	//cv::Scalar colorScalar = cv::Scalar( 94, 206, 165 );
	//rectangle(image, topLeft, bottomRight, colorScalar, thickness, lineType, shift);


	display("orig", image);

	Mat miniMat = image(Rect(topLeft, bottomRight));

	Mat blurred = miniMat;
	GaussianBlur(miniMat, blurred, Size( 5, 5 ), 0, 0 );
	//display("blur", blurred);
	

	/*Mat edges;
	double threshold1 = 1; double threshold2 = 3; int apertureSize=3; bool L2gradient=false;
	Canny(blurred, edges, threshold1, threshold2, apertureSize, L2gradient);
	namedWindow( "canny", WINDOW_AUTOSIZE );
	imshow( "canny", edges );
    waitKey(0); */

	Mat ycbcr;
	cvtColor(blurred,ycbcr,CV_RGB2YCrCb);
	Mat chan[3];
	split(ycbcr,chan);
	Mat y  = chan[0];
	display("Y", y);

	Mat binary;
	cv::threshold(y, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	display("binary", binary); 

	
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
	display("binary1", binary);


	//Mat binrgb;
	//cvtColor(binary, binrgb, CV_GRAY2RGB);
	Mat mask(image.rows, image.cols, CV_8UC1, Scalar(0,0,0));
	//cout << "HERE8" <<endl;
	Mat mask_roi(mask, Rect(topLeft, bottomRight));
	binary.copyTo(mask_roi);
	display("mask", mask); 

	inpaint_image(image, mask);
}

void inpaint_image(Mat image, Mat mask)
{
	Mat inpainted;
    inpaint(image, mask, inpainted, 10, cv::INPAINT_TELEA  );
	display("inpainted", inpainted);
}

int roundPos(double n) //works for positive n only
{
    return  floor(n + 0.5);
}

void display(String s, Mat img)
{
	namedWindow( s, WINDOW_AUTOSIZE );
	imshow( s, img );
	waitKey(0); 
}


/*
cvtColor(source, destination, CV_BGR2Lab);
The pixel values can then be accessed in the following manner:

int step = destination.step;
int channels = destination.channels();
for (int i = 0; i < destination.rows(); i++) {
    for (int j = 0; j < destination.cols(); j++) {
        Point3_<uchar> pixelData;
        //L*: 0-255 (elsewhere is represented by 0 to 100)
        pixelData.x = destination.data[step*i + channels*j + 0];
        //a*: 0-255 (elsewhere is represented by -127 to 127)
        pixelData.y = destination.data[step*i + channels*j + 1];
        //b*: 0-255 (elsewhere is represented by -127 to 127)
        pixelData.z = destination.data[step*i + channels*j + 2];
    }
}
*/


/*double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);        // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}*/