cmake_minimum_required(VERSION 3.10)
project(opencv_ocr_digits)

set(CMAKE_CXX_STANDARD 11)

add_executable(opencv_ocr_digits main.cpp CvThreshold.cpp CvThreshold.h)

find_package( OpenCV REQUIRED )
find_library( OpenCV REQUIRED )

target_link_libraries(opencv_ocr_digits opencv_core opencv_highgui opencv_videoio opencv_imgproc opencv_text)