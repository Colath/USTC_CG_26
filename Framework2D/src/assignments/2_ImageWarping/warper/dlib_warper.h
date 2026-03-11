#pragma once

#include "warper.h"
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4127) 
#include <dlib/image_transforms.h>
#include <dlib/matrix.h>
#pragma warning(pop)

namespace USTC_CG
{

class DlibWarper : public Warper
{
public:
    DlibWarper() = default;
    virtual ~DlibWarper() = default;

    void warp(double x, double y, double& new_x, double& new_y) override;
    
    void set_control_points(
        const std::vector<std::pair<double, double>>& start_points,
        const std::vector<std::pair<double, double>>& end_points) override;

private:
    std::vector<std::pair<double, double>> start_pts_;
    std::vector<std::pair<double, double>> end_pts_;

    dlib::point_transform_affine tform_;
    bool is_ready_ = false;
};

}  // namespace USTC_CG
