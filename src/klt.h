#ifndef _USE_LK_OPTICAL_FLOW_
#define _USE_LK_OPTICAL_FLOW_

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

struct KltTrackerConfig
{
    int patch_size = 21;
    int n_max_iteration = 10;
    int max_pyramid_level = 3;
};

class KltTracker
{
public:
    enum TrackStatus : uint8_t
    {
        kFailed = 0,  // Hard failure: point is permanently lost.
        kSuccess = 1, // Point successfully tracked at current / finest level.
        kSoftFail = 2 // Soft failure (e.g. border exclusion) at a coarse pyramid level; retryable at finer levels.
    };

    KltTracker() = default;
    virtual ~KltTracker() = default;

    void setConfig(const KltTrackerConfig &config) { config_ = config; }

    virtual bool performTracking(const cv::Mat &image_ref, const cv::Mat &image_cur, const Eigen::MatrixXd &gradient_x,
                                 const Eigen::MatrixXd &gradient_y, const std::vector<cv::Point2d> &prev_kp, std::vector<cv::Point2d> &next_kp,
                                 std::vector<uint8_t> &status) = 0;

    virtual bool performPyramidTracking(const cv::Mat &image_ref, const cv::Mat &image_cur, const std::vector<cv::Point2d> &prev_kp,
                                        std::vector<cv::Point2d> &next_kp, std::vector<uint8_t> &status) = 0;

protected:
    void preComputeImageGradient(const cv::Mat &image, cv::Mat &gradient_x, cv::Mat &gradient_y);

    void preComputeImagePyramid(const cv::Mat &image, std::vector<cv::Mat> &image_pyramid);

    inline bool inBorder(const cv::Mat &image, const int rowIndex, const int colIndex, const int step = 0)
    {
        return rowIndex > 0 && rowIndex < image.rows - step && colIndex > 0 && colIndex < image.cols - step;
    }

    double BilinearInterpolation(cv::Mat &mat, const double row, const double col)
    {
        int row_floor = std::floor(row);
        int row_ceil = std::ceil(row);
        int col_floor = std::floor(col);
        int col_ceil = std::ceil(col);

        if (row_floor < 0 || row_ceil >= mat.rows || col_floor < 0 || col_ceil >= mat.cols)
        {
            // Out of bounds, return 0 or some default value. The caller should ideally check bounds before calling this function.
            return 0.0;
        }

        double delta_row = row - row_floor;
        double delta_col = col - col_floor;

        double value = (1 - delta_row) * (1 - delta_col) * mat.at<uchar>(row_floor, col_floor) +
                       (1 - delta_row) * delta_col * mat.at<uchar>(row_floor, col_ceil) +
                       delta_row * (1 - delta_col) * mat.at<uchar>(row_ceil, col_floor) + delta_row * delta_col * mat.at<uchar>(row_ceil, col_ceil);

        return value;
    }

    double BilinearInterpolation(const Eigen::MatrixXd &mat, const double row, const double col)
    {
        int row_floor = std::floor(row);
        int row_ceil = std::ceil(row);
        int col_floor = std::floor(col);
        int col_ceil = std::ceil(col);

        if (row_floor < 0 || row_ceil >= mat.rows() || col_floor < 0 || col_ceil >= mat.cols())
        {
            // Out of bounds, return 0 or some default value. The caller should ideally check bounds before calling this function.
            return 0.f;
        }

        double delta_row = row - row_floor;
        double delta_col = col - col_floor;

        double value = (1 - delta_row) * (1 - delta_col) * mat(row_floor, col_floor) + (1 - delta_row) * delta_col * mat(row_floor, col_ceil) +
                       delta_row * (1 - delta_col) * mat(row_ceil, col_floor) + delta_row * delta_col * mat(row_ceil, col_ceil);

        return value;
    }

    KltTrackerConfig config_;
};

#endif