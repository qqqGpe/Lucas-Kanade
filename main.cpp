#include "src/klt.h"
#include "src/klt_forward.h"
#include "src/klt_inverse.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#define USE_INVERSE_COMPOSITION_TRACKER // Comment out to use forward compositional tracker instead

namespace
{
constexpr int kMaxHarrisCorners = 200;
constexpr double kHarrisQualityLevel = 0.01;
constexpr double kHarrisMinDistance = 20;
constexpr int kKltPatchSize = 21;
constexpr int kKltMaxPyramidLevel = 3;
constexpr int kKltMaxIteration = 10;
} // namespace

int main(int argc, char **argv)
{
    std::string path_to_dataset = std::string(PROJECT_DIR) + "/data";
    std::string associate_file = path_to_dataset + "/associate.txt";
    std::cout << "associate file is: " << associate_file << std::endl;

    std::ifstream fin(associate_file);
    if (!fin)
    {
        std::cerr << "Failed to find associate.txt!" << std::endl;
        exit(1);
    }

    std::string rgb_file, depth_file, time_rgb, time_depth;
    std::vector<cv::Point2d> curr_keypoints;
    std::vector<cv::Point2d> prev_keypoints;
    cv::Mat prev_color_image, curr_color_image;
    cv::Mat prev_gray_image, curr_gray_image;

#ifdef USE_INVERSE_COMPOSITION_TRACKER
    std::unique_ptr<KltTracker> tracker = std::make_unique<KltInverseCompositionTracker>();
#else
    std::unique_ptr<KltTracker> tracker = std::make_unique<KltForwardCompositionTracker>();
#endif
    KltTrackerConfig klt_config{
        .patch_size = kKltPatchSize,
        .n_max_iteration = kKltMaxIteration,
        .max_pyramid_level = kKltMaxPyramidLevel
    };
    tracker->setConfig(klt_config);

    for (int index = 0; index < 9; index++)
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        curr_color_image = cv::imread(path_to_dataset + "/" + rgb_file);
        cv::cvtColor(curr_color_image, curr_gray_image, cv::COLOR_BGR2GRAY);

        if (index == 0)
        {
            std::vector<cv::Point2d> corners;
            cv::goodFeaturesToTrack(curr_gray_image, corners, kMaxHarrisCorners, kHarrisQualityLevel, kHarrisMinDistance);
            prev_keypoints = corners;
            curr_keypoints = corners;
        }
        else
        {
            std::vector<uint8_t> status;
            std::vector<float> error;
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            tracker->performPyramidTracking(prev_gray_image, curr_gray_image, prev_keypoints, curr_keypoints, status);

            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

            const size_t n_tracked = std::count_if(status.begin(), status.end(), [](uint8_t s) { return s != 0; });
            std::cout << "frame " << index << ": tracked " << n_tracked << "/" << prev_keypoints.size() << " keypoints"
                      << ", cost time: " << time_used.count() << " seconds." << std::endl;

            // Show the optical flow
            cv::Mat image_to_show = curr_color_image.clone();
            for (int j = 0; j < prev_keypoints.size(); j++)
            {
                if (status[j])
                {
                    cv::line(image_to_show, prev_keypoints[j], curr_keypoints[j], cv::Scalar(255, 0, 0), 2);
                    cv::circle(image_to_show, curr_keypoints[j], 3, cv::Scalar(0, 0, 255), -1);
                }
            }

            // show image
            cv::imshow("show_klt", image_to_show);
            cv::waitKey(0);
        }

        prev_color_image = curr_color_image.clone();
        prev_gray_image = curr_gray_image.clone();
        prev_keypoints = curr_keypoints;
    }
    return 0;
}