#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

Mat_<float> div(const Mat &image){
  int n = *(image.size + 1);
  int m = *(image.size + 2);
  int pdim[3] = {2, n, m};
  Mat temp(3, pdim, image.type());
  Mat result(n, m, CV_32F);
  for (int i = 0; i < n; i++){
    temp.at<float>(0,0,i) = image.at<float>(0,0,i);
    temp.at<float>(0,n-1,i) = -image.at<float>(0,n-2,i); 
  }
  for (int j = 0; j < m; j++){
    temp.at<float>(1,j,0) = image.at<float>(1,j,0);
    temp.at<float>(1,j,m-1) = -image.at<float>(1,j,m-2);
  }

  for (int i = 1; i < n -1; i++){
    for (int j = 0; j < m; j++){
      temp.at<float>(0,i,j) = image.at<float>(0,i,j) - image.at<float>(0,i-1,j);
    }
  }
  for (int i = 0; i < n; i++){
    for (int j = 1; j < m - 1; j++){
      temp.at<float>(1,i,j) = image.at<float>(1,i,j) - image.at<float>(1,i,j-1); 
    }
  }
  for (int i = 0; i < n; i++){
    for (int j = 1; j < m - 1; j++){
      result.at<float>(i,j) = temp.at<float>(0,i,j) + temp.at<float>(1,i,j);
    }
  }
  return result;
}
Mat_<float> grad(const Mat &image){
  int pdim[3] = {2, image.rows, image.cols};
  Mat result = Mat::zeros(3, pdim, CV_32F);
  for (int i = 0; i < image.rows - 1; i++){
    for (int j = 0; j < image.cols; j++){
      result.at<float>(0,i,j) = image.at<float>(i+1,j) - image.at<float>(i,j);
    }
  }
  for (int i = 0; i < image.rows; i++){
    for (int j = 0; j < image.cols - 1; j++){
      result.at<float>(1,i,j) = image.at<float>(i,j+1) - image.at<float>(i,j);
    }
  }
  return result;
}

Mat_<float> magnitude(const Mat &image){
  int n = *(image.size + 1);
  int m = *(image.size + 2);
  Mat result(n,m,CV_32F);
  for (int i = 0; i < n; i++){
    for (int j = 0; j < m; j++){
      result.at<float>(i,j) = sqrt(image.at<float>(0,i,j)*image.at<float>(0,i,j) + image.at<float>(1,i,j)*image.at<float>(1,i,j));
    }
  }
  return result;
}

Mat_<float> proj_div(const float lam, const Mat &image){
  int n = *(image.size + 1);
  int m = *(image.size + 2);
  Mat mag = magnitude(image);
  mag = max(mag,lam);
  Mat result(image.size(), image.type());
  //for (int i = 0; i < n; i++){
  //  for (int j = 0; j < m; j++){
  //    result.at<float>(0,i,j) = lam * image.at<float>(0,i,j)/mag.at<float>(i,j);
  //    result.at<float>(1,i,j) = lam * image.at<float>(1,i,j)/mag.at<float>(i,j);
  //  }
  //}
  return result;
}

Mat_<float> ChambollePock_Denoise(const float lam, const Mat &image, float tau, float sig, float theta, float tol){
  int pdim[3] = {2,image.rows, image.cols }; // correct order?
  Mat p = Mat::zeros(3,pdim,CV_32F);
  Mat p_hat = Mat::zeros(3,pdim,CV_32F);
  Mat divp = Mat::zeros(image.size(), CV_32F);
  Mat u = Mat::zeros(image.size(), CV_32F);
  Mat u_prev = Mat::zeros(image.size(), CV_32F);
  Mat u_hat = Mat::zeros(image.size(), CV_32F);
  Mat gradu = Mat::zeros(3,pdim,CV_32F);
  divp = div(p);
  gradu = grad(u_hat);
  int maxiter = 100;
  for (int i = 0; i < maxiter; i++){
    u_prev = u.clone();
    p_hat = p + sig * gradu;
    //p = proj_div(lam,p_hat);
    //divp = div(p);
    //u = 1.0/(1.0 + tau) * (u + tau*divp + tau*image);
    //u_hat = u + theta*(u - u_prev);
  }
  return u;
}


int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
    //Mat_<float> fimage(image.size());
    //image.convertTo(fimage, CV_32F, 1/255.0);
    //Mat noise(fimage.size(), fimage.type());
    //randn(noise,0.0,0.1);
    //add(fimage,noise,fimage);
    Mat result = ChambollePock_Denoise(0.5, image, 0.1, 0.1, 1.0, 0.1); 

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", fimage );                   // Show our image inside it.

    //waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
