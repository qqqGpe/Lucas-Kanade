#include "klt.h"

#include <Eigen/Cholesky>

class KltInverseCompositionTracker : public KltTracker
{
public:
    KltInverseCompositionTracker() = default;

    ~KltInverseCompositionTracker() override = default;

    bool performTracking(const cv::Mat &image_ref, const cv::Mat &image_cur, const Eigen::MatrixXd &gradient_x, const Eigen::MatrixXd &gradient_y,
                         const std::vector<cv::Point2d> &prev_kp, std::vector<cv::Point2d> &next_kp, std::vector<uint8_t> &status) override;

    bool performPyramidTracking(const cv::Mat &image_ref, const cv::Mat &image_cur, const std::vector<cv::Point2d> &prev_kp,
                                std::vector<cv::Point2d> &next_kp, std::vector<uint8_t> &status) override;

    void preComputeHessian(const cv::Mat &image, const std::vector<cv::Point2d> &keypoints, Eigen::MatrixXd &hessian);

private:
    constexpr static double kStopThreshold = 1e-3;

    virtual void Compute_dW_dp(const Eigen::Vector2d &uv, const Eigen::VectorXd &p, Eigen::MatrixXd &dW_dp) override;

    virtual void UpdateWarpParameter(Eigen::VectorXd &p, Eigen::VectorXd &dp) override;

    virtual void WarpSinglePoint(const Eigen::Vector2d &point, const Eigen::VectorXd &warp_p, Eigen::Vector2d &warped_point) override;

    void InitializeWarpParameters(const Eigen::Vector2d &prev_p, const Eigen::Vector2d &cur_p, Eigen::VectorXd &p);

    Eigen::MatrixXd preHessian_;
    std::vector<Eigen::LDLT<Eigen::Matrix2d>> preHessianSolver_;
    std::vector<cv::Mat> image_ref_pyramid_;
    std::vector<cv::Mat> image_cur_pyramid_;
    std::vector<Eigen::MatrixXd> preGradient_x_pyramid_;
    std::vector<Eigen::MatrixXd> preGradient_y_pyramid_;
};