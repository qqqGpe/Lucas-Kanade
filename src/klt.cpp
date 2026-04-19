#include "klt.h"

void KltTracker::preComputeImagePyramid(const cv::Mat &image, std::vector<cv::Mat> &image_pyramid)
{
    image_pyramid.clear();
    image_pyramid.push_back(image);
    for (int i = 1; i < config_.max_pyramid_level; i++)
    {
        cv::Mat down_image;
        cv::pyrDown(image_pyramid[i - 1], down_image);
        image_pyramid.push_back(down_image);
    }
}

void KltTracker::preComputeImageGradient(const cv::Mat &image, cv::Mat &gradient_x, cv::Mat &gradient_y)
{
    // 3x3 Sobel sums to 8 on a unit ramp; scale by 1/8 so the values match the true derivative.
    cv::Sobel(image, gradient_x, CV_32F, 1, 0, 3, 1.0 / 8.0);
    cv::Sobel(image, gradient_y, CV_32F, 0, 1, 3, 1.0 / 8.0);
}
