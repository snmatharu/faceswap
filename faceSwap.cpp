#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string> 
using namespace cv;
using namespace std;
vector<Point2f> readPoints(string pointsFileName){
	vector<Point2f> points;
	ifstream ifs (pointsFileName.c_str());
    float x, y;
	int count = 0;
    while(ifs >> x >> y)
    {
        points.push_back(Point2f(x,y));

    }

	return points;
}
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
    Mat warpMat = getAffineTransform( srcTri, dstTri );
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}
static void calculateDelaunayTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri){

    Subdiv2D subdiv(rect);

    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);	        

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point2f> pt(3);
	vector<int> ind(3);

	for( size_t i = 0; i < triangleList.size(); i++ )
	{
		Vec6f t = triangleList[i];
		pt[0] = Point2f(t[0], t[1]);
		pt[1] = Point2f(t[2], t[3]);
		pt[2] = Point2f(t[4], t[5 ]);

		if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
			for(int j = 0; j < 3; j++)
				for(size_t k = 0; k < points.size(); k++)
					if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)						
						ind[j] = k;					

			delaunayTri.push_back(ind);
		}
	}
		
}



void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &t1, vector<Point2f> &t2)
{
    
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);
    
    vector<Point2f> t1Rect, t2Rect;
    vector<Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {

        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
        t2RectInt.push_back( Point(t2[i].x - r2.x, t2[i].y - r2.y) ); 
    }
    
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
 
    Mat img1Rect;
    img1(r1).copyTo(img1Rect);
    
    Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());
    
    applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);
    
    multiply(img2Rect,mask, img2Rect);
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2Rect;
    
    
}


int main( int argc, char** argv)
{	
    string filename1 = "ted_cruz.jpg";
    string filename2 = "donald_trump.jpg";
    
    Mat img1 = imread(filename1);
    Mat img2 = imread(filename2);
    Mat img1Warped = img2.clone();
	
	vector<Point2f> points1, points2;
	points1 = readPoints(filename1 + ".txt");
	points2 = readPoints(filename2 + ".txt");
    
    img1.convertTo(img1, CV_32F);
    img1Warped.convertTo(img1Warped, CV_32F);
    
    
    vector<Point2f> hull1;
    vector<Point2f> hull2;
    vector<int> hullIndex;
    
    convexHull(points2, hullIndex, false, false);
    
    for(int i = 0; i < hullIndex.size(); i++)
    {
        hull1.push_back(points1[hullIndex[i]]);
        hull2.push_back(points2[hullIndex[i]]);
    }

    
    vector< vector<int> > dt;
	Rect rect(0, 0, img1Warped.cols, img1Warped.rows);
	calculateDelaunayTriangles(rect, hull2, dt);
	
	for(size_t i = 0; i < dt.size(); i++)
    {
        vector<Point2f> t1, t2;
        
		for(size_t j = 0; j < 3; j++)
        {
			t1.push_back(hull1[dt[i][j]]);
			t2.push_back(hull2[dt[i][j]]);
		}
        
        warpTriangle(img1, img1Warped, t1, t2);

	}
    
    vector<Point> hull8U;
    for(int i = 0; i < hull2.size(); i++)
    {
        Point pt(hull2[i].x, hull2[i].y);
        hull8U.push_back(pt);
    }

    Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
    fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,255,255));

    Rect r = boundingRect(hull2);
    Point center = (r.tl() + r.br()) / 2;
    
    Mat output;
    img1Warped.convertTo(img1Warped, CV_8UC3);
	seamlessClone(img1Warped,img2, mask, center, output, NORMAL_CLONE);
    
    imshow("Face Swapped", output);
    waitKey(0);
    destroyAllWindows();
    

	return 1;
}
