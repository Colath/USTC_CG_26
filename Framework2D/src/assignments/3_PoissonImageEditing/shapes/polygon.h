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

    // 获取多边形内部的所有像素（扫描线算法核心实现）
    std::vector<std::pair<int, int>> get_interior_pixels() const;

    // 结束绘制
    void finish_drawing();

   private:
    std::vector<std::pair<float, float>> vertices_;
    float current_x_ = 0.0f;
    float current_y_ = 0.0f;
    bool is_finished_ = false;
};
}  // namespace USTC_CG