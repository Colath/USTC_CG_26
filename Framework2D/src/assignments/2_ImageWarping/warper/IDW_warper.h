#pragma once

#include "warper.h"
#include <vector>
#include <Eigen/Dense>

namespace USTC_CG
{
class IDWWarper : public Warper
{
   public:
    IDWWarper() = default;
    virtual ~IDWWarper() = default;
    // HW2_TODO: Implement the warp(...) function with IDW interpolation
    void warp(double x, double y, double& new_x, double& new_y) override;

    // HW2_TODO: other functions or variables if you need
    void set_control_points(const std::vector<std::pair<double, double>>& start_points,
                            const std::vector<std::pair<double, double>>& end_points) override;

   private:
    std::vector<Eigen::Vector2d> p_i;
    std::vector<Eigen::Vector2d> q_i;
    std::vector<Eigen::Matrix2d> T_i;

    void compute_matrices();
    double sigma(double x1, double y1, double x2, double y2) const;
};
}  // namespace USTC_CG