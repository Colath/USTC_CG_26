#include "IDW_warper.h"
#include <cmath>
#include <limits>
#include <iostream>

namespace USTC_CG
{

void IDWWarper::set_control_points(
    const std::vector<std::pair<double, double>>& start_points,
    const std::vector<std::pair<double, double>>& end_points)
{
    p_i.clear();
    q_i.clear();
    T_i.clear();
    
    if (start_points.size() != end_points.size()) return;

    for (size_t i = 0; i < start_points.size(); ++i) {
        p_i.push_back(Eigen::Vector2d(start_points[i].first, start_points[i].second));
        q_i.push_back(Eigen::Vector2d(end_points[i].first, end_points[i].second));
    }

    compute_matrices();
}

double IDWWarper::sigma(double x1, double y1, double x2, double y2) const
{
    double dx = x1 - x2;
    double dy = y1 - y2;
    double dist_sq = dx * dx + dy * dy;
    if (dist_sq < 1e-10) {
        return std::numeric_limits<double>::infinity();
    }
    return 1.0 / dist_sq;
}

void IDWWarper::compute_matrices()
{
    size_t n = p_i.size();
    T_i.resize(n);

    for (size_t i = 0; i < n; ++i) {
        Eigen::Matrix2d A = Eigen::Matrix2d::Zero();
        Eigen::Matrix2d B = Eigen::Matrix2d::Zero();

        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;

            double sig = sigma(p_i[j].x(), p_i[j].y(), p_i[i].x(), p_i[i].y());
            Eigen::Vector2d dp = p_i[j] - p_i[i];
            Eigen::Vector2d dq = q_i[j] - q_i[i];

            // A += sigma * (p_j - p_i) * (p_j - p_i)^T
            A += sig * (dp * dp.transpose());

            // B += sigma * (q_j - q_i) * (p_j - p_i)^T
            B += sig * (dq * dp.transpose());
        }

        if (n < 2) {
            T_i[i] = Eigen::Matrix2d::Identity();
        } else {
            if (std::abs(A.determinant()) < 1e-10) {
                // If A is not invertible, fallback to identity
                T_i[i] = Eigen::Matrix2d::Identity();
            } else {
                T_i[i] = B * A.inverse();
            }
        }
    }
}

void IDWWarper::warp(double x, double y, double& new_x, double& new_y)
{
    if (p_i.empty()) {
        new_x = x;
        new_y = y;
        return;
    }

    double sum_w = 0.0;
    new_x = 0.0;
    new_y = 0.0;
    
    // Check if the point matches any control point exactly
    for (size_t i = 0; i < p_i.size(); ++i) {
        double dist_sq = (x - p_i[i].x()) * (x - p_i[i].x()) + (y - p_i[i].y()) * (y - p_i[i].y());
        if (dist_sq < 1e-10) {
            new_x = q_i[i].x();
            new_y = q_i[i].y();
            return;
        }
    }

    for (size_t i = 0; i < p_i.size(); ++i) {
        double w_i = sigma(x, y, p_i[i].x(), p_i[i].y());

        Eigen::Vector2d p_vec(x, y);
        Eigen::Vector2d dp = p_vec - p_i[i];

        // f_i(p) = q_i + T_i * (p - p_i)
        Eigen::Vector2d f_i = q_i[i] + T_i[i] * dp;

        new_x += w_i * f_i.x();
        new_y += w_i * f_i.y();
        sum_w += w_i;
    }

    if (sum_w > 0.0) {
        new_x /= sum_w;
        new_y /= sum_w;
    } else {
        new_x = x;
        new_y = y;
    }
}

}  // namespace USTC_CG
