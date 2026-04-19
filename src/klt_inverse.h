#include "klt.h"

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
    Eigen::MatrixXd preHessian_;
    std::vector<cv::Mat> image_ref_pyramid_;
    std::vector<cv::Mat> image_cur_pyramid_;
    std::vector<Eigen::MatrixXd> preGradient_x_pyramid_;
    std::vector<Eigen::MatrixXd> preGradient_y_pyramid_;
};