#pragma once

#include <vector>

#include "shape.h"

namespace USTC_CG
{
class Freehand : public Shape
{
   public:
    Freehand() = default;
    ~Freehand() override = default;

    void add_control_point(float x, float y) override;
    void update(float x, float y) override;  // freehand: update == append point
    void draw(const Config& config) const override;

    int num_points() const
    {
        return (int)points_.size();
    }

   private:
    struct Pt
    {
        float x, y;
    };
    std::vector<Pt> points_;

    // in case the sampling points are too dense
    float min_dist_ = 2.0f;
};
}  // namespace USTC_CG
