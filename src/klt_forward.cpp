#include "klt_forward.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <opencv2/imgproc.hpp>

void KltForwardCompositionTracker::Compute_dW_dp(const Eigen::Vector2d &uv, const Eigen::VectorXd &p, Eigen::MatrixXd &dW_dp)
{
    // For translation-only klt, dw/dp is simply Identity matrix
    dW_dp = Eigen::Matrix2d::Identity();
}

void KltForwardCompositionTracker::UpdateWarpParameter(Eigen::VectorXd &p, Eigen::VectorXd &dp)
{
    // For translation-only klt, updating warp param is just an additive operation
    p = p + dp;
}

void KltForwardCompositionTracker::InitializeWarpParameters(const Eigen::Vector2d &prev_p, const Eigen::Vector2d &cur_p, Eigen::VectorXd &p)
{
    // For translation-only klt, initializing warp param is just computing the difference between current keypoint and previous keypoint
    p(0) = cur_p.x() - prev_p.x();
    p(1) = cur_p.y() - prev_p.y();
}

void KltForwardCompositionTracker::WarpSinglePoint(const Eigen::Vector2d &point, const Eigen::VectorXd &warp_parameters,
                                                   Eigen::Vector2d &warped_point)
{
    // For translation-only klt, warping a single point is just adding the translation to the original point
    warped_point = point + warp_parameters;
}

bool KltForwardCompositionTracker::performTracking(const cv::Mat &image_ref, const cv::Mat &image_cur, const Eigen::MatrixXd &gradient_x,
                                                   const Eigen::MatrixXd &gradient_y, const std::vector<cv::Point2d> &prev_kp,
                                                   std::vector<cv::Point2d> &next_kp, std::vector<uint8_t> &status)
{
    const bool has_prediction = !next_kp.empty();
    if (!has_prediction)
    {
        next_kp.resize(prev_kp.size());
    }

    for (int i = 0; i < prev_kp.size(); i++)
    {
        if (status[i] == kFailed)
        {
            continue;
        }

        cv::Mat template_patch;
        const Eigen::Vector2d prev_keypoint(prev_kp[i].x, prev_kp[i].y);
        const int border_margin = config_.patch_size / 2 + 1;
        if (!inBorder(image_ref, prev_keypoint.y(), prev_keypoint.x(), border_margin))
        {
            // Soft fail: template patch out of bounds at this level. At a finer level the same
            // physical point may have enough margin, so don't permanently kill the point here.
            status[i] = kSoftFail;
            continue;
        }
        cv::getRectSubPix(image_ref, cv::Size(config_.patch_size, config_.patch_size), prev_kp[i], template_patch);

        // For translation-only klt, warp param is 2D
        Eigen::VectorXd warp_parameters = Eigen::VectorXd::Zero(2);
        Eigen::VectorXd delta_p = Eigen::VectorXd::Zero(2);
        if (has_prediction)
        {
            const Eigen::Vector2d next_keypoint(next_kp[i].x, next_kp[i].y);
            InitializeWarpParameters(prev_keypoint, next_keypoint, warp_parameters);
        }

        for (int iter_time = 0; iter_time < config_.n_max_iteration; iter_time++)
        {
            cv::Mat warped_patch;
            Eigen::Vector2d warped_point;

            WarpSinglePoint(prev_keypoint, warp_parameters, warped_point);
            if (!inBorder(image_cur, warped_point.y(), warped_point.x(), border_margin))
            {
                status[i] = kSoftFail;
                break;
            }
            cv::getRectSubPix(image_cur, cv::Size(config_.patch_size, config_.patch_size), cv::Point2f(warped_point.x(), warped_point.y()),
                              warped_patch);

            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 2);
            Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
            for (int u = -config_.patch_size / 2; u <= config_.patch_size / 2; u++)
            {
                for (int v = -config_.patch_size / 2; v <= config_.patch_size / 2; v++)
                {
                    Eigen::MatrixXd dW_dp;
                    Compute_dW_dp(warped_point, warp_parameters, dW_dp);

                    double error = static_cast<double>(template_patch.at<uchar>(v + config_.patch_size / 2, u + config_.patch_size / 2) -
                                                       warped_patch.at<uchar>(v + config_.patch_size / 2, u + config_.patch_size / 2));

                    Eigen::MatrixXd gradient_I = Eigen::MatrixXd::Zero(1, 2);
                    double dx = BilinearInterpolation(gradient_x, warped_point.y() + v, warped_point.x() + u);
                    double dy = BilinearInterpolation(gradient_y, warped_point.y() + v, warped_point.x() + u);
                    if (dx == 0 || dy == 0 || std::isnan(dx) || std::isnan(dy))
                    {
                        continue;
                    }

                    gradient_I << dx, dy;
                    H += (gradient_I * dW_dp).transpose() * (gradient_I * dW_dp);
                    b += (gradient_I * dW_dp).transpose() * error;
                }
            }

            delta_p = H.ldlt().solve(b);
            UpdateWarpParameter(warp_parameters, delta_p);
            if (delta_p.norm() < kStopThreshold)
            {
                break;
            }
        }
        Eigen::Vector2d final_warped_point;
        WarpSinglePoint(prev_keypoint, warp_parameters, final_warped_point);
        if (!inBorder(image_cur, final_warped_point.y(), final_warped_point.x()))
        {
            status[i] = kSoftFail;
            continue;
        }
        next_kp[i] = cv::Point2d(final_warped_point.x(), final_warped_point.y());
        status[i] = kSuccess;
    }
    return true;
}

bool KltForwardCompositionTracker::performPyramidTracking(const cv::Mat &image_ref, const cv::Mat &image_cur, const std::vector<cv::Point2d> &prev_kp,
                                                          std::vector<cv::Point2d> &next_kp, std::vector<uint8_t> &status)
{
    image_ref_pyramid_.clear();
    image_cur_pyramid_.clear();
    preGradient_x_pyramid_.clear();
    preGradient_y_pyramid_.clear();

    preComputeImagePyramid(image_ref, image_ref_pyramid_);
    preComputeImagePyramid(image_cur, image_cur_pyramid_);
    for (int i = 0; i < image_ref_pyramid_.size(); i++)
    {
        cv::Mat gradient_x, gradient_y;
        preComputeImageGradient(image_cur_pyramid_[i], gradient_x, gradient_y);
        Eigen::MatrixXd gradient_x_eigen, gradient_y_eigen;
        cv::cv2eigen(gradient_x, gradient_x_eigen);
        cv::cv2eigen(gradient_y, gradient_y_eigen);
        preGradient_x_pyramid_.push_back(gradient_x_eigen);
        preGradient_y_pyramid_.push_back(gradient_y_eigen);
    }

    const size_t n_points = prev_kp.size();
    const int n_levels = config_.max_pyramid_level;
    if (n_levels <= 0)
    {
        std::cerr << "Error: max_pyramid_level should be greater than 0." << std::endl;
        return false;
    }

    const bool has_prediction = (next_kp.size() == n_points);
    status.assign(n_points, kSuccess);

    std::vector<cv::Point2d> prev_kp_scaled(n_points);
    std::vector<cv::Point2d> next_kp_scaled(n_points);

    for (int level = n_levels - 1; level >= 0; level--)
    {
        const double scale = 1.0 / (1 << level);
        const bool is_top_level = (level == n_levels - 1);

        for (size_t i = 0; i < n_points; i++)
        {
            if (status[i] == kFailed)
            {
                continue;
            }
            prev_kp_scaled[i] = prev_kp[i] * scale;
            if (is_top_level)
            {
                // Top level: seed from caller prediction (scaled) or from prev_kp (zero warp).
                next_kp_scaled[i] = has_prediction ? next_kp[i] * scale : prev_kp_scaled[i];
            }
            else if (status[i] == kSoftFail)
            {
                // Previous (coarser) level border-failed this point: at this finer level there
                // may be enough margin, so retry with a clean zero-warp seed.
                next_kp_scaled[i] = prev_kp_scaled[i];
                status[i] = kSuccess;
            }
            else
            {
                // Finer level: upsample previous level's result.
                next_kp_scaled[i] = next_kp_scaled[i] * 2.0;
            }
        }

        performTracking(image_ref_pyramid_[level], image_cur_pyramid_[level], preGradient_x_pyramid_[level], preGradient_y_pyramid_[level],
                        prev_kp_scaled, next_kp_scaled, status);
    }

    // At the finest level any remaining soft-fail becomes a hard fail for the caller.
    for (size_t i = 0; i < n_points; i++)
    {
        if (status[i] == kSoftFail)
        {
            status[i] = kFailed;
        }
    }

    next_kp = next_kp_scaled;
    return true;
}