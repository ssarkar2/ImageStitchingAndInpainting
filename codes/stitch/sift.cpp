#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <iostream>
#include <math.h> 
#include <random>
#include <time.h> 

#define PI 3.14159265
#define DISPLAY 1;

using namespace cv;
using namespace std;

typedef struct keypoint
{
	int x;
	int y;
	double sigma;
	int octave;
	int level;
	int peak;
	double binval;
	double *descriptor;
}keypoint;


void printMtx1(double* mtx, int rows, int cols);
double **matMul(double *mtx1, double *mtx2, int r1, int c1r2, int c2);
void MatrixInversion(double **A, int order, double **Y);
int GetMinor(double **src, double **dest, int row, int col, int order);
double CalcDeterminant( double **mat, int order);
int* choose8(int range);
double* ransac(vector<keypoint> kp1, vector<keypoint> kp2, vector<Point> matches, double ransacTh, int ransacIter);
vector<Point> matchDescriptors(vector<keypoint> kp1, vector<keypoint> kp2, double matchThreshold);
Mat stitchImages(vector<keypoint> kp1, vector<keypoint> kp2, Mat image1, Mat image2, double matchThreshold, double ransacTh, int ransacIter);
int bound(int num, int low, int high);
double* getDescriptor(keypoint* newkeypoint, Mat image, double *sigmavalues, int levelsPerOctave);
double interpolateValue(double x, double y, Mat image);
vector<int> findPeaks(double* anglebin, int length, double peaktolerance);
double gaussian(double sigma, double dist);
double inverseTan(double x, double y);
void displayInterestPoints(Mat image, vector<Vec4f> interestpoints, String s);
vector<keypoint> siftFeatures(Mat image, double sigma, double k, int ksize0, int ksize1, int numOctave, int levelsPerOctave, double DoGthreshold, double eigratio, int neighbourhoodRadius, double peaktolerance);
void display(String s, Mat img);


void test();
//notes
//what if best match (from matchDescriptors) are on different scales?
int main( int argc, char** argv )
{
	double sigma = 1.6, k = sqrt(2.0);
	int numOctave = 3, levelsPerOctave = 6;
	int ksize0 = 0, ksize1 = 0;
	double DoGthreshold = 0.01;
	double eigratio = 10, peaktolerance = 0.8;
	int neighbourhoodRadius = 2;
	double matchThreshold = 0.8;
	double ransacTh = 8; int ransacIter = 200;


	//test();


	Mat image1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	Mat imagefloat1;
	image1.convertTo(imagefloat1, CV_64F, 1.0/255.0); 
#if DISPLAY
	display("origfloat1", imagefloat1);
#endif
	cout << "size : " <<Point(imagefloat1.rows, imagefloat1.cols) <<endl;

	/*circle( image1, Point( 218, 75 ), 5.0, Scalar( 0, 0, 255 ), 1, 8 );
	display("orig1", image1);*/

	vector<keypoint> kp1 = siftFeatures(imagefloat1, sigma, k, ksize0, ksize1, numOctave, levelsPerOctave, DoGthreshold, eigratio, neighbourhoodRadius, peaktolerance);

	Mat image2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE );
	Mat imagefloat2;
	image2.convertTo(imagefloat2, CV_64F, 1.0/255.0); 
#if DISPLAY
	display("origfloat2", imagefloat2);
#endif
	cout << "size : " <<Point(imagefloat2.rows, imagefloat2.cols) <<endl;

	/*circle( image2, Point( 16, 137 ), 5.0, Scalar( 0, 0, 255 ), 1, 8 );
	circle( image2, Point( 162, 163 ), 5.0, Scalar( 0, 0, 255 ), 1, 8 );
	display("orig2", image2);*/

	vector<keypoint> kp2 = siftFeatures(imagefloat2, sigma, k, ksize0, ksize1, numOctave, levelsPerOctave, DoGthreshold, eigratio, neighbourhoodRadius, peaktolerance);
	
	Mat stitched = stitchImages(kp1, kp2, image1, image2, matchThreshold, ransacTh, ransacIter);
	display("stitched", stitched);
}



Mat stitchImages(vector<keypoint> kp1, vector<keypoint> kp2, Mat image1, Mat image2, double matchThreshold, double ransacTh, int ransacIter)
{
	//match keypoints and find closest neighbours
	//select a few and find the transformation
	//transform and paste
	vector<Point> matches = matchDescriptors(kp1, kp2, matchThreshold);
	ransac(kp1, kp2, matches, ransacTh, ransacIter);

	Mat a;
	return a;
}

double* ransac(vector<keypoint> kp1, vector<keypoint> kp2, vector<Point> matches, double ransacTh, int ransacIter)
{
	int *randomPoints;
	double *z = (double*)calloc(8*16 , sizeof(double));

	double* transformation = (double*)calloc(9, sizeof(double));
	double* bestmodel = (double*)calloc(9, sizeof(double));
	
	double *b = (double*)calloc(16 , sizeof(double));
	double point[3]; double **tempans;
	Point match, P1, P2;
	double xcomponent, ycomponent, error;
	double minerr = 10000000000000000000000000000000.0;
	Mat Z, pinvZ, B, H;
	for (int iters = 0; iters < ransacIter; iters++)
	{
		randomPoints = choose8(matches.size());

		//build homography matrix
		cout << "iters: " << iters <<endl;
		for (int i = 0; i < 16; i+=2)
		{
			//cout << "i: " << i <<endl;
			/*if (i == 8)
			{
				cout <<"crash"<<endl;
			}*/
			//cout << "randompt: " << randomPoints[i>>1] <<endl;
			match = matches[randomPoints[i>>1]];
			//cout << "match" << match <<endl;
			P1 = Point(kp1[match.x].x, kp1[match.x].y);
			P2 = Point(kp2[match.y].x, kp2[match.y].y);
			cout << "P1 " << P1 << "   P2 " << P2<< endl;


			z[i*8 + 0] = P1.x; 
			z[i*8 + 1] = P1.y;
			z[i*8 + 2] = 1;
			z[i*8 + 6] = -P1.x * P2.x;
			z[i*8 + 7] = -P1.y * P2.x;
			z[(i+1)*8 + 3] = P1.x; 
			z[(i+1)*8 + 4] = P1.y;
			z[(i+1)*8 + 5] = 1;
			z[(i+1)*8 + 6] = -P1.x * P2.y;
			z[(i+1)*8 + 7] = -P1.y * P2.y;
			b[i] = P2.x;
			b[i+1] = P2.y;

		}

		Z = Mat(16, 8, CV_64FC1, z);
		pinvZ = Z.inv(DECOMP_SVD );
		B = Mat(16, 1, CV_64FC1, b);
		H = pinvZ * B;
		//cout << H <<endl;


		transformation[0] = H.at<double>(0,0);
		transformation[1] = H.at<double>(1,0);
		transformation[2] = H.at<double>(2,0);
		transformation[3] = H.at<double>(3,0);
		transformation[4] = H.at<double>(4,0);
		transformation[5] = H.at<double>(5,0);
		transformation[6] = H.at<double>(6,0);
		transformation[7] = H.at<double>(7,0);
		transformation[8] = 1;

		//printMtx1(transformation, 3, 3);

		error = 0;
		for (int i = 0; i < matches.size(); i++)
		{
			//cout << "errorcalc : " << i << "/" << matches.size() << endl;
			point[0] = kp1[matches[i].x].x;
			point[1] = kp1[matches[i].x].y;
			point[2] = 1;
			tempans = matMul(transformation, point, 3, 3, 1);
			tempans[0][0] = tempans[0][0]/tempans[2][0];
			tempans[1][0] = tempans[1][0]/tempans[2][0];

			//calculate error
			xcomponent = (tempans[0][0] - kp2[matches[i].y].x); ycomponent = (tempans[1][0] - kp2[matches[i].y].y);
			error += sqrt(xcomponent*xcomponent + ycomponent*ycomponent);
			free(tempans);
		}
		if (error < minerr)
		{
			minerr = error;
			for (int mm = 0; mm < 9; mm++)
			{
				bestmodel[mm] = transformation[mm];
			}
		}
		if (error > 10000000000000000000000000000000.0)
		{
			cout <<"trouble" <<endl;
		}
		free(randomPoints);
	}
	printMtx1(bestmodel, 3, 3);
	return bestmodel;
}

void printMtx1(double* mtx, int rows, int cols)
{
	cout <<endl;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cout << mtx[i*cols + j] << "  " ;
		}
		cout <<endl;
	}
}

double **matMul(double *mtx1, double *mtx2, int r1, int c1r2, int c2)
{
	double **product = (double**)calloc(r1 , sizeof(double*));
	for (int i = 0; i < r1; i++)
	{
		product[i] = (double*)calloc(c2 , sizeof(double));
	}

	double sum = 0;
	for (int i = 0; i < r1; i++) //ith row of mtx 1
	{
		for (int j = 0; j < c2; j++)  //jth column of mtx 2
		{
			sum = 0;
			for (int k = 0; k < c1r2; k++)  //kth element of the dot product
			{
				//cout << mtx1[i*c1r2 + k] << " " << mtx2[k*c2 + j] << endl ; 
				sum += mtx1[i*c1r2 + k]*mtx2[k*c2 + j];
			}
			product[i][j] = sum;
			//cout << "sum "<<product[i][j] << "  " << sum <<endl; 
		}
	}
	return product;
}

//choose 8 integers in [0, range)
int* choose8(int range)
{
	int* numlist = (int*)calloc(8, sizeof(int));
	int temp, count = 0;
	vector<int> usedlist;
	if (range < 8)
		cout << "trouble: not enough elements" <<endl;

	std::default_random_engine generator (time(0));

	std::uniform_int_distribution<> d(1, range);
	while(1)
	{
		temp = d(generator);
		//cout << "xx" << temp <<endl;
		if(std::find(usedlist.begin(), usedlist.end(), temp) != usedlist.end())
		{
			//element already present. do nothing
		}
		else
		{
			numlist[count] = temp-1;
			count++;
		}
		if (count >= 8)
			break;
	}
	return numlist;
}

vector<Point> matchDescriptors(vector<keypoint> kp1, vector<keypoint> kp2, double matchThreshold)
{
	double sumsq = 0; double min = 0, min2 = 0;
	int bestmatch = 0;
	vector<Point> bestMaches;
	for (int i = 0; i < kp1.size(); i++)
	{
		min = 1000000000000000000000000000000000000.0;
		min2 = min;
		for (int j = 0; j < kp2.size(); j++)
		{
			sumsq = 0;
			bestmatch = 0;
			for (int m = 0; m < 128; m++)//loop over the 128 points in the feature
			{
				//cout << kp1[i].descriptor[m] << " " <<  kp2[j].descriptor[m] << " " << kp1[i].x << " " << kp2[j].x <<" "<<  kp1[i].y << " " << kp2[j].y <<endl;
				sumsq += (kp1[i].descriptor[m] - kp2[j].descriptor[m]) * (kp1[i].descriptor[m] - kp2[j].descriptor[m]);
			}
			if (min > sqrt(sumsq))
			{
				min2 = min;
				min = sqrt(sumsq);
				bestmatch = j;
			}
			//cout << "j: " << j << " sqrt(sumsq) " << sqrt(sumsq) << endl;
		}
		if (min/min2 < matchThreshold)
			bestMaches.push_back(Point(i, bestmatch));
	}

	/*for (int i = 0; i < bestMaches.size(); i++)
	{

		cout << bestMaches[i] <<endl;
	}*/

	return bestMaches;
}

vector<keypoint> siftFeatures(Mat image, double sigma, double k, int ksize0, int ksize1, int numOctave, int levelsPerOctave, double DoGthreshold, double eigratio, int neighbourhoodRadius, double peaktolerance)
{
	Mat temp1, temp2, diffimg, temp, downimage;
	Mat *pmat = (Mat*)malloc(sizeof(Mat));
	int countMat = 0, countDiff = 0;  //number of elements in gaussFiltered
	vector<Mat> gaussFiltered;
	vector<Mat> DoG;
	double s = sigma;
	Mat imageWorking = image.clone();
	vector<Mat> imageScales;
	double DoGx, DoGy, DoGs, DoGxx, DoGyy, DoGss, DoGxy, DoGxs, DoGys;
	double traceH, detH;
	double det00, det01, det02, det10, det11, det12, det20, det21, det22;
	double optimalx, optimaly, optimals;

	double *deltaX = (double*)malloc(sizeof(double) * neighbourhoodRadius*neighbourhoodRadius);  //check if required
	double *deltaY = (double*)malloc(sizeof(double) * neighbourhoodRadius*neighbourhoodRadius); //check if required

	double *anglebin = (double*)calloc(36, sizeof(double));  //36 bins of angles

	double deltax, deltay, mag, angle;
	int binindex;

	double *sigmavalues = (double*)malloc(sizeof(double) * numOctave*levelsPerOctave);
	double currsigma, currdist;

	int numkeypoints = 0, correcti, correctj;
	vector<keypoint> keypointlist;
	keypoint* newkeypoint;
	for (int countOctave = 0; countOctave < numOctave; countOctave++)
	{
		s = sigma * pow(k*k, countOctave);
		//display("imageWorking",imageWorking );
		imageScales.push_back(imageWorking.clone());
		for (int countLevels = 0; countLevels < levelsPerOctave ; countLevels++)
		{
			s = s * k;
			cout << "octave = " << countOctave << " level = " << countLevels << " sigma = " << s <<endl;
			*(sigmavalues + countOctave*levelsPerOctave + countLevels) = s;
			GaussianBlur(imageWorking, temp, Size(ksize0, ksize1), s);
 			gaussFiltered.push_back(temp.clone());
			//display("temp",temp);
			countMat++;
			if (countLevels >= 1)
			{
				temp1 = gaussFiltered[countMat-1];
				temp2 = gaussFiltered[countMat-2];
				//display("temp1", temp1);
				//display("temp2", temp2);
				subtract(temp1, temp2, diffimg);
				DoG.push_back(diffimg.clone());
				countDiff++;
				//equalizeHist( DoG[countDiff-1] , diffimg );
				//display("diff", diffimg);
			}
		}		
		resize(imageWorking, imageWorking, Size(0,0), 0.5, 0.5, INTER_NEAREST);
	}

	//find max/min points that are possible keypoints
	int rowstart, rowend, colstart, colend, startscale, endscale;
	int maxcandidate, mincandidate;
	double currpointvalue;
	vector<Vec4f> candidateinterestpoints;
	vector<Vec4f> filteredinterestpoints;
	for (int traverseOctaves = 0; traverseOctaves < numOctave; traverseOctaves++)  //Check later: numOctave instead of 1
	{
		for(int level = 1; level < levelsPerOctave-2; level++)
		{
			cout << "level: " << level<<endl << endl;
			for (int i = neighbourhoodRadius+1; i < DoG[traverseOctaves*(levelsPerOctave-1)].rows - (neighbourhoodRadius+1); i++) //row
			{
				
				for (int j = neighbourhoodRadius+1; j < DoG[traverseOctaves*(levelsPerOctave-1)].cols - (neighbourhoodRadius+1); j++) //col
				{
					rowstart = i-1;
					rowend = i+1;
					colstart = j-1;
					colend = j+1;
					startscale = level - 1;
					endscale = level + 1;

					maxcandidate = 1; mincandidate = 1;
					currpointvalue = DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i,j);

					if (std::abs(currpointvalue) < DoGthreshold)
						continue;

					//check the neighbouring 26 points
					for (int scalecounter = startscale; scalecounter <= endscale; scalecounter++)
					{
						for (int checkneighbours_row = rowstart ; checkneighbours_row <= rowend ; checkneighbours_row++)
						{
							for (int checkneighbours_col = colstart ; checkneighbours_col <= colend ; checkneighbours_col++)
							{
								if (DoG[traverseOctaves*(levelsPerOctave-1) + scalecounter].at<double>(checkneighbours_row, checkneighbours_col) > currpointvalue)  //its (levelsPerOctave-1) because if there are 4 levels then there are 3 differences
									maxcandidate = 0;
								if (DoG[traverseOctaves*(levelsPerOctave-1) + scalecounter].at<double>(checkneighbours_row, checkneighbours_col) < currpointvalue)
									mincandidate = 0;
								if (maxcandidate == 0 && mincandidate == 0)
									break;
							}
							if (maxcandidate == 0 && mincandidate == 0)
								break;
						}
						if (maxcandidate == 0 && mincandidate == 0)
							break;
					}
					
					if (maxcandidate == 1 || mincandidate == 1) //we have a possible candidate
					{
						candidateinterestpoints.push_back(Vec4f(i, j, level, traverseOctaves));
						//calculate 1st and 2nd derivatives (hessian) of the DoG approximation of LoG
						DoGx = (DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i,j+1) - DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i,j-1)) * 0.5;
						DoGy = (DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+1,j) - DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i-1,j)) * 0.5;
						DoGs = (DoG[traverseOctaves*(levelsPerOctave-1) + level+1].at<double>(i,j) - DoG[traverseOctaves*(levelsPerOctave-1) + level-1].at<double>(i,j)) * 0.5;
						DoGxx = DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i,j-1) - 2*DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i,j) + DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i,j+1);
						DoGyy = DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i-1,j) - 2*DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i,j) + DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+1,j);
						DoGss = DoG[traverseOctaves*(levelsPerOctave-1) + level-1].at<double>(i,j) - 2*DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i,j) + DoG[traverseOctaves*(levelsPerOctave-1) + level+1].at<double>(i,j);
						DoGxy = (DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+1,j+1) + DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i-1,j-1) - DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+1,j-1) - DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i-1,j+1)) * 0.25;
						DoGxs = (DoG[traverseOctaves*(levelsPerOctave-1) + level+1].at<double>(i,j+1) + DoG[traverseOctaves*(levelsPerOctave-1) + level-1].at<double>(i,j-1) - DoG[traverseOctaves*(levelsPerOctave-1) + level+1].at<double>(i,j-1) - DoG[traverseOctaves*(levelsPerOctave-1) + level-1].at<double>(i,j+1)) * 0.25;
						DoGys = (DoG[traverseOctaves*(levelsPerOctave-1) + level+1].at<double>(i+1,j) + DoG[traverseOctaves*(levelsPerOctave-1) + level-1].at<double>(i-1,j) - DoG[traverseOctaves*(levelsPerOctave-1) + level+1].at<double>(i-1,j) - DoG[traverseOctaves*(levelsPerOctave-1) + level-1].at<double>(i+1,j)) * 0.25;
						
						//hessian is : [DoGxx DoGxy DoGxs; DoGxy DoGyy DoGys; DoGxs DoGys DoGss];
						traceH = DoGxx + DoGyy + DoGss;  //trace of hessian
						detH = (DoGxx*DoGyy*DoGss) + 2 * (DoGxy*DoGxs*DoGys) - (DoGxx*DoGys*DoGys) - (DoGss*DoGxy*DoGxy) - (DoGyy*DoGxs*DoGxs);  //determinant of hessian
						//edge response elimination
						if (traceH*traceH / detH < ((eigratio+1)*(eigratio+1))/(eigratio))
						{
							det00 = DoGyy*DoGss - DoGys*DoGys;
							det01 = DoGxy*DoGss - DoGxs*DoGys;
							det02 = DoGxy*DoGys - DoGyy*DoGxs;
							det10 = DoGxy*DoGss - DoGys*DoGxs;
							det11 = DoGxx*DoGss - DoGxs*DoGxs;
							det12 = DoGxx*DoGys - DoGxs*DoGxy;
							det20 = DoGxy*DoGys - DoGyy*DoGxs;
							det21 = DoGxx*DoGys - DoGxs*DoGxy;
							det22 = DoGxx*DoGyy - DoGxy*DoGxy;
							//the inverse of H is (1/detH) * [det00 -det10 det20;
															//-det01 det11 -det21 ; 
															//det02 -det12 det22]
							//H^-1 * D
							optimalx = (1/detH) * (det00 * DoGx - det10 * DoGy + det20 * DoGs);
							optimaly = (1/detH) * (-det01 * DoGx + det11 * DoGy - det21 * DoGs);
							optimals = (1/detH) * (det02 * DoGx - det12 * DoGy + det22 * DoGs);

							filteredinterestpoints.push_back(Vec4f(i, j, level, traverseOctaves));

							for (int m = 0; m < 2*neighbourhoodRadius+1; m++)  //row
							{
								for (int n = 0; n < 2*neighbourhoodRadius+1; n++)  //col
								{
									//*(deltaX + m*neighbourhoodRadius + n) = DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+m,j+n+1) - DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+m,j+n-1);
									//*(deltaY + m*neighbourhoodRadius + n) = DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+m+1,j+n) - DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+m-1,j+n);
									//mag = 

									if (j+n-1-neighbourhoodRadius < 0 || i+m-1-neighbourhoodRadius < 0 || j+n+1-neighbourhoodRadius >= image.cols || i+m+1-neighbourhoodRadius >= image.rows) 
									{
										cout << "j+n-1   " << j+n-1 << "i+m-1   " << i+m-1 << "j+n+1   " << j+n+1 << "i+m+1   " << i+m+1 <<endl;
										//safe.... not getting prints here
									}
									deltax = DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+m-neighbourhoodRadius,j+n-neighbourhoodRadius + 1) - DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+m-neighbourhoodRadius,j+n-neighbourhoodRadius - 1);
									deltay = DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+m-neighbourhoodRadius + 1,j+n-neighbourhoodRadius) - DoG[traverseOctaves*(levelsPerOctave-1) + level].at<double>(i+m-neighbourhoodRadius - 1,j+n-neighbourhoodRadius);
									mag = sqrt(deltax*deltax + deltay*deltay);
									angle = inverseTan(deltax, deltay);
									binindex = ceil(angle/10.0);
									currsigma = (*(sigmavalues + traverseOctaves*levelsPerOctave + level)) * 1.5;
									currdist = sqrt(double((m - neighbourhoodRadius)*(m - neighbourhoodRadius) + (n - neighbourhoodRadius)*(n - neighbourhoodRadius)));
									anglebin[binindex] = anglebin[binindex] + mag*gaussian(currsigma, currdist);
								}
							}
							vector<int> peaks = findPeaks(anglebin, 36, peaktolerance);
							for (int m = 0; m < peaks.size(); m++)
							{
								correcti = i + optimaly;  //row
								correctj = j + optimalx;  //col
								//save keypoint
								numkeypoints = numkeypoints + 1;
								newkeypoint = (keypoint*)malloc(sizeof(keypoint));
								newkeypoint->x = correctj;
								newkeypoint->y = correcti;
								newkeypoint->sigma = *(sigmavalues + traverseOctaves*levelsPerOctave + level);
								newkeypoint->octave = traverseOctaves;
								newkeypoint->level = level;
								newkeypoint->peak = peaks[m];
								newkeypoint->binval = anglebin[peaks[m]];
								newkeypoint->descriptor = getDescriptor(newkeypoint, DoG[traverseOctaves*(levelsPerOctave-1) + level], sigmavalues, levelsPerOctave);
								keypointlist.push_back(*newkeypoint);
							}
						}//end of if (traceH*traceH / detH < ((eigratio+1)*(eigratio+1)/eigratio))
					}//end of if (maxcandidate == 1 || mincandidate == 1)
				}//end of j
			}//end of i
		}//end of levels
	}//end of octaves

#if DISPLAY
	displayInterestPoints(image.clone(), candidateinterestpoints, "all");
	displayInterestPoints(image.clone(), filteredinterestpoints, "filtered");
#endif
	return keypointlist;
}

double* getDescriptor(keypoint* newkeypoint, Mat image, double *sigmavalues, int levelsPerOctave)
{
	double angleInRad = (newkeypoint->peak)*PI/18;
	double neighbourhood[16][16];
	double cosAngle = cos(angleInRad), sinAngle = sin(angleInRad);
	double x, y;
	int blockstartrow, blockstartcol;
	double deltax, deltay, mag, angle;
	int binindex;
	int nrows = image.rows, ncols = image.cols;
	double *anglebin = (double*)calloc(8, sizeof(double));  //8 bins of angles
	double *feature = (double*)calloc(128, sizeof(double));
	int featurecount = 0;
	double sigma, dist, rowtemp, coltemp;
	double sqsum = 0, sqsum1 = 0;

	for (int i = -8; i <= 7 ; i++) //x
	{
		for (int j = -8; j <= 7 ; j++)//y
		{
			x = cosAngle * i - sinAngle * j + newkeypoint->x;
			y = sinAngle * i + cosAngle * j + newkeypoint->y;
			neighbourhood[i+8][j+8] = interpolateValue(x, y, image);  //find interpolated value
		}
	}

	//now divide neighbourhood into 4x4 regions and find descriptors on each of them
	for (int i = 0; i < 4; i++) //row
	{
		for (int j = 0; j < 4; j++) //col
		{
			blockstartrow = 4*i; blockstartcol = 4*j;  //topleft of current block
			//get current block and calculate its histogram
			for (int p = 0; p < 4; p++) //row
			{
				for (int q = 0; q < 4; q++) //col
				{
					deltax = neighbourhood[bound(blockstartrow + p, 0, nrows)][bound(blockstartcol + q + 1, 0, ncols)] - neighbourhood[bound(blockstartrow + p, 0, nrows)][bound(blockstartcol + q - 1, 0, ncols)];
					deltay = neighbourhood[bound(blockstartrow + p + 1, 0, nrows)][bound(blockstartcol + q, 0, ncols)] - neighbourhood[bound(blockstartrow + p - 1, 0, nrows)][bound(blockstartcol + q, 0, ncols)];
					mag = sqrt(deltax*deltax + deltay*deltay);
					angle = inverseTan(deltax, deltay);
					sigma = (*(sigmavalues + (newkeypoint->octave)*levelsPerOctave + newkeypoint->level)) * 1.5;
					rowtemp = (blockstartrow + p - 8);
					coltemp = (blockstartcol + q - 8);
					dist = sqrt((rowtemp*rowtemp) + (coltemp*coltemp));  //distance wrt centre of the 16x16 block
					binindex = ceil(angle/45.0);   //dividing into 8 bins
					anglebin[binindex] = anglebin[binindex] + mag*gaussian(sigma, dist);
				}
			}

			for (int k = 0; k < 8; k++)
			{
				feature[featurecount] = anglebin[k];
				sqsum += anglebin[k]*anglebin[k]; 
				featurecount++;
			}
		}
	}

	//normalize feature
	for (int k = 0; k < 128; k++)
	{
		feature[k] = feature[k]/sqrt(sqsum);
		if (feature[k] > 0.2)
			feature[k] = 0.2; //saturate
		sqsum1 += feature[k]*feature[k];
	}
	for (int k = 0; k < 128; k++)
	{
		feature[k] = feature[k]/sqrt(sqsum1);
	}

	return feature;
}

int bound(int num, int low, int high)
{
	int ans = num;
	if (ans < low)
		ans = low;
	if (ans >= high)
		ans = high;
	return ans;
}

double interpolateValue(double x, double y, Mat image)
{
	int xleft = floor(x), xright = ceil(x), ytop = floor(y), ybottom = ceil(y);  //integer coordinates for the 4 corners based on which we will interpolate the value at x,y
	if (x < 0)
	{
		xleft = 0; xright = 0;
	}
	if (y < 0)
	{
		ytop = 0; ybottom = 0;
	}
	if (x >= image.cols)
	{
		xleft = image.cols - 1; xright = image.cols - 1;
	}
	if (y >= image.rows)
	{
		ytop = image.rows - 1; ybottom = image.rows - 1;
	}

	//the four corners are: (xleft, ytop), (xright, ytop), (xleft, ybottom), (xright, ybottom)
	double lefttop = image.at<double>(ytop, xleft);
	double righttop = image.at<double>(ytop, xright);
	double leftbottom = image.at<double>(ybottom, xleft);
	double rightbottom = image.at<double>(ybottom, xright);

	double xratio = x - xleft, yratio = y - ytop;  //the ratios for division

	//do the interpolation in 3 steps
	double interp1 = xratio * righttop + (1-xratio) * lefttop;
	double interp2 = xratio * rightbottom + (1-xratio) * leftbottom;
	return (1-yratio) * interp1 + yratio * interp2;
}

vector<int> findPeaks(double* anglebin, int length, double peaktolerance)
{
	vector<int> peakindices;
	double max = -1000000; int maxloc = 0;
	for (int i = 0; i < length; i++)
	{
		if (anglebin[i] > max)
		{
			max = anglebin[i]; maxloc = i;
		}
	}
	for (int i = 0; i < length; i++)
	{
		if (anglebin[i] > peaktolerance*max)
		{
			peakindices.push_back(i);
		}
	}
	return peakindices;
}

double gaussian(double sigma, double dist)
{
	return exp(-(dist*dist)/(2*sigma*sigma))/(sigma*sqrt(2*PI));
}


double inverseTan(double x, double y)
{
	double angle180 = (atan2(y, x) * 180 / PI); //in -180 to 180
	if (angle180 < 0)
		return 360.0 + angle180;
	else
		angle180;
}

void displayInterestPoints(Mat image, vector<Vec4f> interestpoints, String s)
{
	for(int i = 0; i < interestpoints.size(); i++)
		circle(image, Point(interestpoints[i][1], interestpoints[i][0]), interestpoints[i][2]+1, Scalar(0.5));
	cout << "number of points = " << interestpoints.size() <<endl;
	display(s, image);
}

void display(String s, Mat img)
{
	namedWindow( s, WINDOW_AUTOSIZE );
	imshow( s, img );
	waitKey(0); 
}

//matrix inversion code from here: https://chi3x10.wordpress.com/2008/05/28/calculate-matrix-inversion-in-c/
void MatrixInversion(double **A, int order, double **Y)
{
	cout << "mtx inv " <<endl;
    // get the determinant of a
    double det = 1.0/CalcDeterminant(A,order);
 
    // memory allocation
    double *temp = new double[(order-1)*(order-1)];
    double **minor = new double*[order-1];
    for(int i=0;i<order-1;i++)
        minor[i] = temp+(i*(order-1));
 
    for(int j=0;j<order;j++)
    {
		cout << "mtxinv j: " << j <<endl;
        for(int i=0;i<order;i++)
        {
            // get the co-factor (matrix) of A(j,i)
            GetMinor(A,minor,j,i,order);
            Y[i][j] = det*CalcDeterminant(minor,order-1);
            if( (i+j)%2 == 1)
                Y[i][j] = -Y[i][j];
        }
    }
 
    // release memory
    //delete [] minor[0];
    delete [] temp;
    delete [] minor;
}
 
// calculate the cofactor of element (row,col)
int GetMinor(double **src, double **dest, int row, int col, int order)
{
    // indicate which col and row is being copied to dest
    int colCount=0,rowCount=0;
 
    for(int i = 0; i < order; i++ )
    {
        if( i != row )
        {
            colCount = 0;
            for(int j = 0; j < order; j++ )
            {
                // when j is not the element
                if( j != col )
                {
                    dest[rowCount][colCount] = src[i][j];
                    colCount++;
                }
            }
            rowCount++;
        }
    }
 
    return 1;
}
 
// Calculate the determinant recursively.
double CalcDeterminant( double **mat, int order)
{
    // order must be >= 0
    // stop the recursion when matrix is a single element
    if( order == 1 )
        return mat[0][0];
 
    // the determinant value
    double det = 0;
 
    // allocate the cofactor matrix
    double **minor;
    minor = new double*[order-1];
    for(int i=0;i<order-1;i++)
        minor[i] = new double[order-1];
 
    for(int i = 0; i < order; i++ )
    {
		cout << "CalcDeterminant: i: " << i << " order: "<< order <<endl;
        // get minor of element (0,i)
        GetMinor( mat, minor, 0, i , order);
        // the recusion is here!
 
        det += (i%2==1?-1.0:1.0) * mat[0][i] * CalcDeterminant(minor,order-1);
        //det += pow( -1.0, i ) * mat[0][i] * CalcDeterminant( minor,order-1 );
    }
 
    // release memory
    for(int i=0;i<order-1;i++)
        delete [] minor[i];
    delete [] minor;
 
    return det;
}

void test()
{
	double *z = (double*)calloc(8*16 , sizeof(double));
	double **homTxhom, **pinv;

	double *b = (double*)calloc(16 , sizeof(double));
	Point match, P1, P2;
	Mat Z, pinvZ, B, H;
	double* transformation = (double*)calloc(9, sizeof(double));

	vector<Point> ptvector;
	ptvector.push_back(Point(0,0));
	ptvector.push_back(Point(0,0));//1
	ptvector.push_back(Point(2,0));
	ptvector.push_back(Point(0,2));//2
	ptvector.push_back(Point(0,2));
	ptvector.push_back(Point(-2,0));//3
	ptvector.push_back(Point(-2,0));
	ptvector.push_back(Point(0,-2));//4
	ptvector.push_back(Point(0,-2));
	ptvector.push_back(Point(2,0));//5
	ptvector.push_back(Point(3,3));
	ptvector.push_back(Point(-3,3));//6
	ptvector.push_back(Point(-3,3));
	ptvector.push_back(Point(-3,-3));//7
	ptvector.push_back(Point(5,-5));
	ptvector.push_back(Point(5,5));//8

	for (int i = 0; i < 16; i+=2)
		{
			P1 = ptvector[i];
			P2 = ptvector[i+1];
			
			z[i*8 + 0] = P1.x; 
			z[i*8 + 1] = P1.y;
			z[i*8 + 2] = 1;
			z[i*8 + 6] = -P1.x * P2.x;
			z[i*8 + 7] = -P1.y * P2.x;
			z[(i+1)*8 + 3] = P1.x; 
			z[(i+1)*8 + 4] = P1.y;
			z[(i+1)*8 + 5] = 1;
			z[(i+1)*8 + 6] = -P1.x * P2.y;
			z[(i+1)*8 + 7] = -P1.y * P2.y;
			b[i] = P2.x;
			b[i+1] = P2.y;
		}

		Z = Mat(16, 8, CV_64FC1, z);
		pinvZ = Z.inv(DECOMP_SVD );
		B = Mat(16, 1, CV_64FC1, b);
		H = pinvZ * B;
		cout << H <<endl;


		transformation[0] = H.at<double>(0,0);
		transformation[1] = H.at<double>(1,0);
		transformation[2] = H.at<double>(2,0);
		transformation[3] = H.at<double>(3,0);
		transformation[4] = H.at<double>(4,0);
		transformation[5] = H.at<double>(5,0);
		transformation[6] = H.at<double>(6,0);
		transformation[7] = H.at<double>(7,0);
		transformation[8] = 1;

		printMtx1(transformation, 3, 3);
		
}