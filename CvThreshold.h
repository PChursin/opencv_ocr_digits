#ifndef _THRESHOLDER
#define _THRESHOLDER
#include <cv.h>

using namespace cv;

enum class CvThresholdMethod{OTSU,NIBLACK,SAUVOLA,WOLFJOLION};


class CvThreshold
{
public :
    void doThreshold(InputArray src ,OutputArray dst,const CvThresholdMethod &method);
private:
};

#endif //_THRESHOLDER