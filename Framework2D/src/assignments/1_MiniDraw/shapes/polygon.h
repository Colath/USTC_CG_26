#pragma once

#include <vector>

#include "shape.h"

namespace USTC_CG
{
class Polygon : public Shape
{
   public:
    Polygon() = default;
    virtual ~Polygon() = default;

    void finish()
    {
        finished_ = true;
        has_preview_ = false;
    }

    int num_points() const
    {
        return (int)points_.size();
    }

    // add points by clicking
    void add_control_point(float x, float y) override;

    void update(float x, float y) override;

    void draw(const Config& config) const override;

    // helpers
    bool empty() const
    {
        return points_.empty();
    }
    size_t size() const
    {
        return points_.size();
    }

   private:
    struct Pt
    {
        float x, y;
    };

    std::vector<Pt> points_;

    // preview point (mouse position) when drawing polygon
    bool has_preview_ = false;
    bool finished_ = false;

    Pt preview_{ 0.f, 0.f };
};
}  // namespace USTC_CG
