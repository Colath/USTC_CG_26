#include "RBF_warper.h"
#include <cmath>
#include <iostream>
#include <limits>

namespace USTC_CG
{

void RBFWarper::set_control_points(
    const std::vector<std::pair<double, double>>& start_points,
    const std::vector<std::pair<double, double>>& end_points)
{
    p_i.clear();
    q_i.clear();
    r_i.clear();

    if (start_points.size() != end_points.size()) return;

    for (size_t i = 0; i < start_points.size(); ++i) {
        p_i.push_back(Eigen::Vector2d(start_points[i].first, start_points[i].second));
        q_i.push_back(Eigen::Vector2d(end_points[i].first, end_points[i].second));
    }

    compute_coefficients();
}

double RBFWarper::g(double d, double r) const
{
    return std::sqrt(d * d + r * r);
}

void RBFWarper::compute_coefficients()
{
    size_t n = p_i.size();
    if (n == 0) {
        A = Eigen::Matrix2d::Identity();
        b.setZero();
        alpha = Eigen::MatrixXd::Zero(0, 2);
        return;
    }

    r_i.resize(n);
    for (size_t i = 0; i < n; ++i) {
        double min_d = std::numeric_limits<double>::max();
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double d = (p_i[i] - p_i[j]).norm();
            if (d < min_d) min_d = d;
        }
        r_i[i] = (n > 1) ? min_d : 1.0;
        if (r_i[i] < 1e-5) r_i[i] = 1.0;
    }

    if (n < 3) {
        A = Eigen::Matrix2d::Identity();
        if (n == 1) {
            b = q_i[0] - p_i[0];
        } else if (n == 2) {
            b = (q_i[0] - p_i[0] + q_i[1] - p_i[1]) / 2.0;
        }
        alpha = Eigen::MatrixXd::Zero(n, 2);
        return;
    }

    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n + 3, n + 3);
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n + 3, 2);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double d = (p_i[i] - p_i[j]).norm();
            L(i, j) = g(d, r_i[j]);
        }
        L(i, n) = p_i[i].x();
        L(i, n + 1) = p_i[i].y();
        L(i, n + 2) = 1.0;

        L(n, i) = p_i[i].x();
        L(n + 1, i) = p_i[i].y();
        L(n + 2, i) = 1.0;

        rhs(i, 0) = q_i[i].x();
        rhs(i, 1) = q_i[i].y();
    }

    Eigen::MatrixXd sol = L.colPivHouseholderQr().solve(rhs);

    alpha = sol.block(0, 0, n, 2);
    A(0, 0) = sol(n, 0);     A(0, 1) = sol(n + 1, 0);
    A(1, 0) = sol(n, 1);     A(1, 1) = sol(n + 1, 1);
    b(0) = sol(n + 2, 0);
    b(1) = sol(n + 2, 1);
}

void RBFWarper::warp(double x, double y, double& new_x, double& new_y)
{
    Eigen::Vector2d p(x, y);
    Eigen::Vector2d res = A * p + b;

    size_t n = p_i.size();
    for (size_t i = 0; i < n; ++i) {
        double d = (p - p_i[i]).norm();
        double w = g(d, r_i[i]);
        res.x() += alpha(i, 0) * w;
        res.y() += alpha(i, 1) * w;
    }

    new_x = res.x();
    new_y = res.y();
}

}  // namespace USTC_CG
