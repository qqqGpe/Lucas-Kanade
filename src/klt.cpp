#include "klt.h"

inline bool InBorder(cv::Mat image, int rowIndex, int colIndex)
{
    return rowIndex > 0 && rowIndex < image.rows - 1 && colIndex > 0 && colIndex < image.cols - 1;
}

void myCalcOpticalFlowLK(const cv::Mat image1,
                         const cv::Mat image2,
                         std::vector<cv::Point2f>& prev_kp,
                         std::vector<cv::Point2f>& next_kp,
                         std::vector<uint32_t>& status,
                         int patch_size,
                         int max_iteration,
                         bool do_prediction)
{
    constexpr float kStopThreshold = 0.05;
    constexpr float kMaxPointsToTrack = 200;

    if (next_kp.empty())
    {
        if (do_prediction)
        {
            std::cerr << "ERROR, next_kp must not be zero!" << std::endl;
            return;
        }
        next_kp.resize(prev_kp.size());
    }
    assert(next_kp.size() == prev_kp.size());

    if (prev_kp.size() > kMaxPointsToTrack)
    {
        prev_kp.resize(kMaxPointsToTrack);
        next_kp.resize(kMaxPointsToTrack);
    }

    for (int i = 0; i < prev_kp.size(); i++)
    {
        Eigen::Vector2f p = Eigen::Vector2f::Zero();
        if (do_prediction)
        {
            p(0) = next_kp[i].x - prev_kp[i].x;
            p(1) = next_kp[i].y - prev_kp[i].y;
        }

        const int u = prev_kp[i].x;
        const int v = prev_kp[i].y;
        Eigen::Vector2f delta_p = Eigen::Vector2f::Zero();
        Eigen::MatrixXf Jacobian = Eigen::MatrixXf::Zero(1, 2);

        for (int iter = 0; iter <= max_iteration; iter++)
        {
            Eigen::Matrix2f Hessian = Eigen::Matrix2f::Zero();
            Eigen::Vector2f b = Eigen::Vector2f::Zero();

            for (int u_step = -patch_size / 2; u_step < patch_size / 2; u_step++)
            {
                for (int v_step = -patch_size / 2; v_step < patch_size / 2; v_step++)
                {
                    if (!InBorder(image1, v + v_step, u + u_step) || !InBorder(image2, v + v_step + p(1), u + u_step + p(0)))
                    {
                        break;
                    }

                    float jacob_x = 0.5 * (image2.at<uchar>(cv::Point(u + u_step + p(0) + 1, v + v_step + p(1))) -
                                           image2.at<uchar>(cv::Point(u + u_step + p(0) - 1, v + v_step + p(1))));
                    float jacob_y = 0.5 * (image2.at<uchar>(cv::Point(u + u_step + p(0), v + v_step + p(1) + 1)) -
                                           image2.at<uchar>(cv::Point(u + u_step + p(0), v + v_step + p(1) - 1)));
                    Jacobian << jacob_x, jacob_y;

                    const float err = image1.at<uchar>(cv::Point(u + u_step, v + v_step)) - image2.at<uchar>(cv::Point(u + u_step + p(0), v + v_step + p(1)));
                    b += Jacobian.transpose() * err;
                    Hessian += Jacobian.transpose() * Jacobian;
                }
            }

            delta_p = Hessian.ldlt().solve(b);
            p += delta_p;

            if (delta_p.norm() < kStopThreshold)
            {
                break;
            }
        }

        next_kp[i] = cv::Point2f(u + p(0), v + p(1));
    }

    status.resize(prev_kp.size(), 1);
}
