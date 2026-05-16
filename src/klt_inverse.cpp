#include "klt_inverse.h"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <opencv2/imgproc.hpp>

void KltInverseCompositionTracker::Compute_dW_dp(const Eigen::Vector2d &uv, const Eigen::VectorXd &p, Eigen::MatrixXd &dW_dp)
{
    // For translation-only KLT, dW/dp is the identity at every p.
    dW_dp = Eigen::Matrix2d::Identity();
}

void KltInverseCompositionTracker::UpdateWarpParameter(Eigen::VectorXd &p, Eigen::VectorXd &dp)
{
    // Inverse compositional update: W(x; p) <- W(x; p) o W(x; dp)^{-1}.
    // For translation, the inverse of W(x; dp) is W(x; -dp), so the update is p <- p - dp.
    p = p - dp;
}

void KltInverseCompositionTracker::InitializeWarpParameters(const Eigen::Vector2d &prev_p, const Eigen::Vector2d &cur_p, Eigen::VectorXd &p)
{
    p(0) = cur_p.x() - prev_p.x();
    p(1) = cur_p.y() - prev_p.y();
}

void KltInverseCompositionTracker::WarpSinglePoint(const Eigen::Vector2d &point, const Eigen::VectorXd &warp_parameters,
                                                   Eigen::Vector2d &warped_point)
{
    warped_point = point + warp_parameters;
}

void KltInverseCompositionTracker::preComputeHessian(const cv::Mat &image, const std::vector<cv::Point2d> &keypoints, Eigen::MatrixXd &hessian)
{
    // The inverse-compositional Hessian depends only on the template and dW/dp at p = 0,
    // so it stays constant across iterations and can be factored once per keypoint.
    // We stack each 2x2 Hessian vertically into a (2N x 2) matrix.
    cv::Mat grad_x_mat, grad_y_mat;
    preComputeImageGradient(image, grad_x_mat, grad_y_mat);
    Eigen::MatrixXd grad_x, grad_y;
    cv::cv2eigen(grad_x_mat, grad_x);
    cv::cv2eigen(grad_y_mat, grad_y);

    const int n_points = static_cast<int>(keypoints.size());
    const int half = config_.patch_size / 2;
    const int border_margin = half + 1;
    hessian = Eigen::MatrixXd::Zero(2 * n_points, 2);
    // Cache the LDLT factor per keypoint so the inner loop only does back-substitution.
    preHessianSolver_.assign(n_points, Eigen::LDLT<Eigen::Matrix2d>());

    const Eigen::VectorXd p_zero = Eigen::VectorXd::Zero(2);
    Eigen::MatrixXd dW_dp_dyn; // reused storage for the virtual Compute_dW_dp output
    for (int i = 0; i < n_points; i++)
    {
        const Eigen::Vector2d kp(keypoints[i].x, keypoints[i].y);
        if (!inBorder(image, kp.y(), kp.x(), border_margin))
        {
            continue;
        }

        // Translation-only: dW/dp is independent of (u, v), so the virtual call is hoisted
        // outside the pixel loop. For non-translation warps, move it back inside and pass
        // the per-pixel uv.
        Compute_dW_dp(kp, p_zero, dW_dp_dyn);
        const Eigen::Matrix2d dW_dp = dW_dp_dyn;

        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        for (int u = -half; u <= half; u++)
        {
            for (int v = -half; v <= half; v++)
            {
                double dx = BilinearInterpolation(grad_x, kp.y() + v, kp.x() + u);
                double dy = BilinearInterpolation(grad_y, kp.y() + v, kp.x() + u);
                if (dx == 0 || dy == 0 || std::isnan(dx) || std::isnan(dy))
                {
                    continue;
                }

                const Eigen::RowVector2d gradient_T(dx, dy);
                const Eigen::RowVector2d SD = gradient_T * dW_dp;
                H.noalias() += SD.transpose() * SD;
            }
        }

        hessian.block<2, 2>(2 * i, 0) = H;
        preHessianSolver_[i].compute(H);
    }
}

bool KltInverseCompositionTracker::performTracking(const cv::Mat &image_ref, const cv::Mat &image_cur, const Eigen::MatrixXd &gradient_x,
                                                   const Eigen::MatrixXd &gradient_y, const std::vector<cv::Point2d> &prev_kp,
                                                   std::vector<cv::Point2d> &next_kp, std::vector<uint8_t> &status)
{
    const bool has_prediction = !next_kp.empty();
    if (!has_prediction)
    {
        next_kp.resize(prev_kp.size());
    }

    const int n_points = static_cast<int>(prev_kp.size());

    // performPyramidTracking populates preHessian_ / preHessianSolver_ ahead of each level. If this
    // routine is invoked standalone, lazily build them so callers don't have to know the contract.
    if (static_cast<int>(preHessianSolver_.size()) != n_points)
    {
        preComputeHessian(image_ref, prev_kp, preHessian_);
    }

    const int half = config_.patch_size / 2;
    const int border_margin = half + 1;

    const Eigen::VectorXd p_zero = Eigen::VectorXd::Zero(2);
    for (int i = 0; i < n_points; i++)
    {
        if (status[i] == kFailed)
        {
            continue;
        }

        const Eigen::Vector2d prev_keypoint(prev_kp[i].x, prev_kp[i].y);
        if (!inBorder(image_ref, prev_keypoint.y(), prev_keypoint.x(), border_margin))
        {
            status[i] = kSoftFail;
            continue;
        }

        cv::Mat template_patch;
        cv::getRectSubPix(image_ref, cv::Size(config_.patch_size, config_.patch_size), prev_kp[i], template_patch);

        const Eigen::LDLT<Eigen::Matrix2d> &H_solver = preHessianSolver_[i];

        Eigen::VectorXd warp_parameters = Eigen::VectorXd::Zero(2);
        Eigen::VectorXd delta_p = Eigen::VectorXd::Zero(2);
        if (has_prediction)
        {
            const Eigen::Vector2d next_keypoint(next_kp[i].x, next_kp[i].y);
            InitializeWarpParameters(prev_keypoint, next_keypoint, warp_parameters);
        }

        // Translation-only: dW/dp doesn't depend on (u, v) or the current p, so the virtual
        // call is hoisted outside the iteration and pixel loops. Stored as a stack Matrix2d
        // so the per-pixel SD = gradient_T * dW_dp avoids any heap traffic.
        Eigen::MatrixXd dW_dp_dyn;
        Compute_dW_dp(prev_keypoint, p_zero, dW_dp_dyn);
        const Eigen::Matrix2d dW_dp = dW_dp_dyn;

        Eigen::VectorXd b(2); // hoisted: allocated once per keypoint, zeroed each iteration

        for (int iter_time = 0; iter_time < config_.n_max_iteration; iter_time++)
        {
            Eigen::Vector2d warped_point;
            WarpSinglePoint(prev_keypoint, warp_parameters, warped_point);
            if (!inBorder(image_cur, warped_point.y(), warped_point.x(), border_margin))
            {
                status[i] = kSoftFail;
                break;
            }

            cv::Mat warped_patch;
            cv::getRectSubPix(image_cur, cv::Size(config_.patch_size, config_.patch_size), cv::Point2f(warped_point.x(), warped_point.y()),
                              warped_patch);

            b.setZero();
            for (int u = -half; u <= half; u++)
            {
                for (int v = -half; v <= half; v++)
                {
                    // Sample gradient on the TEMPLATE at template coordinates — this is what
                    // keeps the steepest-descent images independent of the current warp p.
                    double dx = BilinearInterpolation(gradient_x, prev_keypoint.y() + v, prev_keypoint.x() + u);
                    double dy = BilinearInterpolation(gradient_y, prev_keypoint.y() + v, prev_keypoint.x() + u);
                    if (dx == 0 || dy == 0 || std::isnan(dx) || std::isnan(dy))
                    {
                        continue;
                    }

                    // error = I(W(x; p)) - T(x), Baker-Matthews convention paired with the p <- p - dp update.
                    const double error = static_cast<double>(warped_patch.at<uchar>(v + half, u + half)) -
                                         static_cast<double>(template_patch.at<uchar>(v + half, u + half));

                    const Eigen::RowVector2d gradient_T(dx, dy);
                    const Eigen::RowVector2d SD = gradient_T * dW_dp;
                    b.noalias() += SD.transpose() * error;
                }
            }

            delta_p = H_solver.solve(b);
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

bool KltInverseCompositionTracker::performPyramidTracking(const cv::Mat &image_ref, const cv::Mat &image_cur, const std::vector<cv::Point2d> &prev_kp,
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
        // Inverse composition takes the gradient on the TEMPLATE (image_ref), not the
        // current image. That's the property that lets the Hessian and steepest-descent
        // images be precomputed once per level.
        cv::Mat gradient_x, gradient_y;
        preComputeImageGradient(image_ref_pyramid_[i], gradient_x, gradient_y);
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
                // Previous (coarser) level border-failed this point: retry at this finer level
                // with a clean zero-warp seed.
                next_kp_scaled[i] = prev_kp_scaled[i];
                status[i] = kSuccess;
            }
            else
            {
                // Finer level: upsample previous level's result.
                next_kp_scaled[i] = next_kp_scaled[i] * 2.0;
            }
        }

        // Precompute per-keypoint Hessians for this level. This is the inverse-compositional
        // win: the Hessian doesn't change across iterations, so it lives outside the inner loop.
        preComputeHessian(image_ref_pyramid_[level], prev_kp_scaled, preHessian_);

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
