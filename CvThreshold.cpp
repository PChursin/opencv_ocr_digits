//
// Created by pavel on 16.07.18.
//

#include "CvThreshold.h"
#include <iostream>
#include <opencv/cv.hpp>

#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;

// *************************************************************
// glide a window across the image and
// create two maps: mean and standard deviation.
// *************************************************************
//#define BINARIZEWOLF_VERSION  "2.3 (February 26th, 2013)"


double calcLocalStats (Mat &im, Mat &map_m, Mat &map_s, int win_x, int win_y) {

    double m,s,max_s, sum, sum_sq, foo;
    int wxh = win_x / 2;
    int wyh = win_y / 2;
    int x_firstth = wxh;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    double winarea = win_x*win_y;

    max_s = 0;
    for (int j = y_firstth ; j<=y_lastth; j++)
    {
        // Calculate the initial window at the beginning of the line
        sum = sum_sq = 0;
        for (int wy=0 ; wy<win_y; wy++)
            for (int wx=0 ; wx<win_x; wx++) {
                foo = im.uget(wx,j-wyh+wy);
                sum    += foo;
                sum_sq += foo*foo;
            }
        m  = sum / winarea;
        s  = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
        if (s > max_s)
            max_s = s;
        map_m.fset(x_firstth, j, m);
        map_s.fset(x_firstth, j, s);

        // Shift the window, add and remove new/old values to the histogram
        for (int i=1 ; i <= im.cols  -win_x; i++) {

            // Remove the left old column and add the right new column
            for (int wy=0; wy<win_y; ++wy) {
                foo = im.uget(i-1,j-wyh+wy);
                sum    -= foo;
                sum_sq -= foo*foo;
                foo = im.uget(i+win_x-1,j-wyh+wy);
                sum    += foo;
                sum_sq += foo*foo;
            }
            m  = sum / winarea;
            s  = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
            if (s > max_s)
                max_s = s;
            map_m.fset(i+wxh, j, m);
            map_s.fset(i+wxh, j, s);
        }
    }

    return max_s;
}




void NiblackSauvolaWolfJolion (InputArray _src, OutputArray _dst,const CvThresholdMethod &version,int winx, int winy, double k, double dR) {

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    double m, s, max_s;
    double th=0;
    double min_I, max_I;
    int wxh = winx/2;
    int wyh = winy/2;
    int x_firstth= wxh;
    int x_lastth = src.cols-wxh-1;
    int y_lastth = src.rows-wyh-1;
    int y_firstth= wyh;
    int mx, my;

    // Create local statistics and store them in a double matrices
    Mat map_m = Mat::zeros (src.size(), CV_32FC1);
    Mat map_s = Mat::zeros (src.size(), CV_32FC1);
    max_s = calcLocalStats (src, map_m, map_s, winx, winy);

    minMaxLoc(src, &min_I, &max_I);

    Mat thsurf (src.size(), CV_32FC1);

    // Create the threshold surface, including border processing
    // ----------------------------------------------------

    for (int j = y_firstth ; j<=y_lastth; j++) {

        // NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
        for (int i=0 ; i <= src.cols-winx; i++) {

            m  = map_m.fget(i+wxh, j);
            s  = map_s.fget(i+wxh, j);

            // Calculate the threshold
            switch (version) {

                case CvThresholdMethod::NIBLACK:
                    th = m + k*s;
                    break;

                case CvThresholdMethod::SAUVOLA:
                    th = m * (1 + k*(s/dR-1));
                    break;

                case CvThresholdMethod::WOLFJOLION:
                    th = m + k * (s/max_s-1) * (m-min_I);
                    break;

                default:
                    std::cerr << "Unknown threshold type in ImageThresholder::surfaceNiblackImproved()\n";
                    exit (1);
            }

            thsurf.fset(i+wxh,j,th);

            if (i==0) {
                // LEFT BORDER
                for (int i=0; i<=x_firstth; ++i)
                    thsurf.fset(i,j,th);

                // LEFT-UPPER CORNER
                if (j==y_firstth)
                    for (int u=0; u<y_firstth; ++u)
                        for (int i=0; i<=x_firstth; ++i)
                            thsurf.fset(i,u,th);

                // LEFT-LOWER CORNER
                if (j==y_lastth)
                    for (int u=y_lastth+1; u<src.rows; ++u)
                        for (int i=0; i<=x_firstth; ++i)
                            thsurf.fset(i,u,th);
            }

            // UPPER BORDER
            if (j==y_firstth)
                for (int u=0; u<y_firstth; ++u)
                    thsurf.fset(i+wxh,u,th);

            // LOWER BORDER
            if (j==y_lastth)
                for (int u=y_lastth+1; u<src.rows; ++u)
                    thsurf.fset(i+wxh,u,th);
        }

        // RIGHT BORDER
        for (int i=x_lastth; i<src.cols; ++i)
            thsurf.fset(i,j,th);

        // RIGHT-UPPER CORNER
        if (j==y_firstth)
            for (int u=0; u<y_firstth; ++u)
                for (int i=x_lastth; i<src.cols; ++i)
                    thsurf.fset(i,u,th);

        // RIGHT-LOWER CORNER
        if (j==y_lastth)
            for (int u=y_lastth+1; u<src.rows; ++u)
                for (int i=x_lastth; i<src.cols; ++i)
                    thsurf.fset(i,u,th);
    }
    //std::cerr << "surface created" << std::endl;


    for (int y=0; y<src.rows; ++y)
        for (int x=0; x<src.cols; ++x)
        {
            if (src.uget(x,y) >= thsurf.fget(x,y))
            {
                dst.uset(x,y,255);
            }
            else
            {
                dst.uset(x,y,0);
            }
        }
}

void CvThreshold::doThreshold(InputArray _src ,OutputArray _dst,const CvThresholdMethod &method)
{
    Mat src = _src.getMat();

    int winx = 0;
    int winy = 0;
    float optK=0.5;
    if (winx==0 || winy==0) {
        winy = (int) (2.0 * src.rows - 1)/3;
        winx = (int) src.cols-1 < winy ? src.cols-1 : winy;

        // if the window is too big, than we assume that the image
        // is not a single text box, but a document page: set
        // the window size to a fixed constant.
        if (winx > 100)
            winx = winy = 40;
    }

    // Threshold
    _dst.create(src.size(), CV_8UC1);
    Mat dst = _dst.getMat();

    //medianBlur(src,dst,5);
    cv::GaussianBlur(src,dst,Size(5,5),0);
//#define _BH_SHOW_IMAGE
#ifdef _BH_DEBUG
#define _BH_SHOW_IMAGE
#endif
    //medianBlur(src,dst,7);
    switch (method)
    {
        case CvThresholdMethod::OTSU :
            threshold(dst,dst,128,255,CV_THRESH_OTSU);
            break;
        case CvThresholdMethod::SAUVOLA :
        case CvThresholdMethod::WOLFJOLION :
            NiblackSauvolaWolfJolion (src, dst, method, winx, winy, optK, 128);
    }

    bitwise_not(dst,dst);


#ifdef _BH_SHOW_IMAGE

#undef _BH_SHOW_IMAGE
#endif
}