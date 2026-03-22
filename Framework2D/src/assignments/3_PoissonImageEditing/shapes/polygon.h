#pragma once

#include <utility>
#include <vector>

#include "shape.h"

namespace USTC_CG
{
class Polygon : public Shape
{
   public:
    Polygon() = default;
    virtual ~Polygon() = default;

    void draw(const Config& config) const override;
    void update(float x, float y) override;
    void add_control_point(float x, float y) override;

    std::vector<std::pair<int, int>> get_interior_pixels() const;

    void finish_drawing();

   private:
    std::vector<std::pair<float, float>> vertices_;
    float current_x_ = 0.0f;
    float current_y_ = 0.0f;
    bool is_finished_ = false;
};
}  // namespace USTC_CG