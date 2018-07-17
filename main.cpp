#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/imgproc/types_c.h>
#include <opencv/cv.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/text/ocr.hpp"
#include "CvThreshold.h"

#define KEY_UP 82
#define KEY_DOWN 84
#define KEY_LEFT 81
#define KEY_RIGHT 83
#define KEY_A 97
#define KEY_D 100
#define KEY_S 115
#define KEY_W 119
using namespace cv;

bool cmpRect(cv::Rect const & lhs, cv::Rect const & rhs) {
    return lhs.tl().x < rhs.tl().x;
};

void usage(char* arg) {
    std::cout<< "Usage:" << std::endl;
    std::cout<< "  "<< arg << " <path to image | 0>" << std::endl;
    std::cout<< "  You can enter '0' as path to read webcam stream" << std::endl;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        usage(argv[0]);
        return 0;
    }
    bool webcam = std::string(argv[1]) == "0";
    Ptr<text::OCRTesseract> tesseract = text::OCRTesseract::create(NULL, "eng", "0123456789", cv::text::OEM_DEFAULT,
            cv::text::PSM_SINGLE_CHAR);
//    Ptr<text::OCRTesseract> tesseract = text::OCRTesseract::create(NULL, "eng", NULL, cv::text::OEM_DEFAULT,
//            cv::text::PSM_SINGLE_CHAR);
//    Ptr<text::OCRTesseract> tesseract = text::OCRTesseract::create(NULL, "rus");
    VideoCapture cap(0);
    Mat frame;
    CvThreshold myThreshold;
    double wCoeff = .25;
    double hCoeff = .25;
    double step = .03;
    double xOffset = 0, yOffset = 0;
    while (true)
    {
        if (webcam)
            cap >> frame;
        else
            frame = imread(argv[1]);
        int w = frame.size().width;
        int h = frame.size().height;
        Rect cutRect(Point(wCoeff*w + xOffset,hCoeff*h + yOffset),
                     Point(w-wCoeff*w + xOffset, h-hCoeff*h + yOffset));

        rectangle(frame, cutRect, Scalar(255, 0, 0));
        Mat cut = frame(cutRect).clone();
        Mat gs_rgb(cut.size(), CV_8UC1);
        //cvtColor(cut, gs_rgb, CV_RGB2GRAY);
        cvtColor(cut, gs_rgb, CV_BGR2GRAY);

        Mat adp(cut.size(), CV_8UC1);
        //cvAdaptiveThreshold(&gs_rgb, &adp, 1);
        //threshold(gs_rgb, adp, 100, 255, THRESH_BINARY_INV);
        myThreshold.doThreshold(gs_rgb, adp, CvThresholdMethod::OTSU);
//        threshold(cut, adp, 100, 255, THRESH_BINARY_INV);

        imshow("Orig", frame);
        imshow("CutGray", gs_rgb);
        Mat margMat = adp(Rect(Point(2, 2), Point(adp.size().width-2, adp.size().height-2))).clone();

        std::vector<std::vector<Point>> cPoints;
        std::vector<Vec4i> hierarchy;
        std::vector<Rect> rects;
        std::vector<Rect> resRects;
        findContours(margMat, cPoints, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        Mat drawing = Mat::zeros(margMat.size(), CV_8UC3);
        for (int i = 0; i < cPoints.size(); i++) {
            drawContours(drawing, cPoints, i, Scalar(255, 0, 0), 2, 8);
            Rect bounds = boundingRect(cPoints[i]);
            rects.push_back(bounds);
//            rectangle(drawing, bounds, Scalar(0, 255, 0), 2);
        }
        std::sort(rects.begin(), rects.end(), cmpRect);
        for (int i = 0; i < rects.size(); i++) {
            Rect cur = rects[i];
            /*for (int j = i+1; j < rects.size(); j++) {
                Rect next = rects[j];
                if (cur.contains(next.tl()) && cur.contains(next.br())) {
                    cur = next;
                    //i++;
                }
            }*/
            resRects.push_back(cur);
        }
        for (Rect r : resRects) {
            rectangle(drawing, r, Scalar(0, 255, 0), 2);
            Mat rectMat = margMat(r).clone();
            std::string res;
            tesseract->run(rectMat, res);
            std::cout << res.c_str() << " ";
        }
        std::cout << std::endl;
        imshow("contours", drawing);
        imshow("CutThres", margMat);
        std::string res;
        tesseract->run(margMat, res);

        std::string noSpaces(res);
//        std::remove_copy(res.begin(), res.end(), std::back_inserter(noSpaces), ' ');

        if (!noSpaces.empty())
            std::cout << "block: " << noSpaces.c_str() << std::endl;
        int ch = waitKey(webcam ? 30 : 0);
        if (ch == 27)
            break;
        else
            switch (ch)
            {
                case KEY_UP:
                    if (hCoeff < .46)
                        hCoeff += step;
                    break;
                case KEY_DOWN:
                    if (hCoeff > 0.05)
                        hCoeff -= step;
                    break;
                case KEY_LEFT:
                    if (wCoeff < .46)
                        wCoeff += step;
                    break;
                case KEY_RIGHT:
                    if (wCoeff > 0.05)
                        wCoeff -= step;
                    break;
                case KEY_A:
                    xOffset -= 5;
                    break;
                case KEY_D:
                    xOffset += 5;
                    break;
                case KEY_W:
                    yOffset -= 5;
                    break;
                case KEY_S:
                    yOffset += 5;
                    break;
                default:
                    std::cout << "KEY: " << ch << std::endl;
                    break;
            }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}