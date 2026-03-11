#include "dlib_warper.h"
#include <iostream>

namespace USTC_CG
{
void DlibWarper::set_control_points(
    const std::vector<std::pair<double, double>>& start_points,
    const std::vector<std::pair<double, double>>& end_points)
{
    start_pts_ = start_points;
    end_pts_ = end_points;
    
    std::vector<dlib::dpoint> from_pts, to_pts;
    for (const auto& p : start_points) from_pts.emplace_back(p.first, p.second);
    for (const auto& p : end_points) to_pts.emplace_back(p.first, p.second);

    if (from_pts.size() >= 3) {
        tform_ = dlib::find_affine_transform(from_pts, to_pts);
        is_ready_ = true;
    } else if (from_pts.size() == 2) {
        tform_ = dlib::find_similarity_transform(from_pts, to_pts);
        is_ready_ = true;
    } else {
        is_ready_ = false;
    }
}

void DlibWarper::warp(double x, double y, double& new_x, double& new_y)
{
    if (is_ready_)
    {
        dlib::dpoint dest = tform_(dlib::dpoint(x, y));
        new_x = dest.x();
        new_y = dest.y();
    }
    else
    {
        new_x = x;
        new_y = y;
    }
}
} // namespace USTC_CG
