#include <chrono>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include "klt.h"

int main(int argc, char** argv)
{
    std::string path_to_dataset = "/home/gao/ws/KLT_tracking/data";
    std::string associate_file = path_to_dataset + "/associate.txt";
    std::cout << "associate file is: " << associate_file << std::endl;

    std::ifstream fin(associate_file);
    if (!fin)
    {
        std::cerr << "Failed to find associate.txt!" << std::endl;
        exit(1);
    }

    std::string rgb_file, depth_file, time_rgb, time_depth;
    // std::vector<cv::Point2f> keypoints;
    std::vector<cv::Point2f> curr_keypoints;
    std::vector<cv::Point2f> prev_keypoints;
    cv::Mat prev_color_image, curr_color_image;
    cv::Mat prev_gray_image, curr_gray_image;
    cv::Mat depth;

    for (int index = 0; index < 9; index++)
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        curr_color_image = cv::imread(path_to_dataset + "/" + rgb_file);
        cv::cvtColor(curr_color_image, curr_gray_image, cv::COLOR_BGR2GRAY);

        if (index == 0)
        {
            std::vector<cv::Point2f> corners;
            int maxCorners = 200;
            double qualityLevel = 0.01;
            double minDistance = 20;
            cv::goodFeaturesToTrack(curr_gray_image, corners, maxCorners, qualityLevel, minDistance);

            prev_keypoints = corners;
            for (int j = 0; j < prev_keypoints.size(); j++)
            {
                cv::circle(curr_gray_image, prev_keypoints[j], 3, cv::Scalar(0, 0, 255), -1);
            }
            cv::imshow("test", curr_gray_image);
            cv::waitKey(0);

            prev_color_image = curr_color_image.clone();
            prev_gray_image = curr_gray_image.clone();
            continue;
        }

        std::vector<uint32_t> status;
        std::vector<float> error;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        myCalcOpticalFlowLK(prev_gray_image, curr_gray_image, prev_keypoints, curr_keypoints, status, 13, 5, false);
        // cv::calcOpticalFlowPyrLK( last_image, image, prev_keypoints, curr_keypoints, status, error, cv::Size(21, 21), 3); // OpenCV klt
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "LK Flow cost time: " << time_used.count() << " seconds." << std::endl;

        cv::circle(prev_color_image, cv::Point2f(100, 200), 5, cv::Scalar(0, 255, 0), -1);

        // show the optical flow
        for (int j = 0; j < prev_keypoints.size(); j++)
        {
            if (status[j])
            {
                cv::line(prev_color_image, prev_keypoints[j], curr_keypoints[j], cv::Scalar(255, 0, 0), 2);
                cv::circle(prev_color_image, curr_keypoints[j], 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // show image
        cv::imshow("show_klt", prev_color_image);
        cv::waitKey(0);

        prev_color_image = curr_color_image.clone();
        prev_gray_image = curr_gray_image.clone();
        prev_keypoints = curr_keypoints;
    }
    return 0;
}