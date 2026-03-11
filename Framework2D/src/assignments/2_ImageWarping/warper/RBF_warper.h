// HW2_TODO: Implement the RBFWarper class
#pragma once

#include "warper.h"
#include <vector>
#include <Eigen/Dense>

namespace USTC_CG
{
class RBFWarper : public Warper
{
   public:
    RBFWarper() = default;
    virtual ~RBFWarper() = default;
    // HW2_TODO: Implement the warp(...) function with RBF interpolation
    void warp(double x, double y, double& new_x, double& new_y) override;

    // HW2_TODO: other functions or variables if you need
    void set_control_points(const std::vector<std::pair<double, double>>& start_points,
                            const std::vector<std::pair<double, double>>& end_points) override;

   private:
    std::vector<Eigen::Vector2d> p_i;
    std::vector<Eigen::Vector2d> q_i;
    std::vector<double> r_i;

    Eigen::MatrixXd alpha;
    Eigen::Matrix2d A;
    Eigen::Vector2d b;

    void compute_coefficients();
    double g(double d, double r) const;
};
}  // namespace USTC_CG