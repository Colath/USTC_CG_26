#include "freehand.h"

#include <imgui.h>

#include <cmath>

namespace USTC_CG
{
static float dist2(float ax, float ay, float bx, float by)
{
    float dx = ax - bx, dy = ay - by;
    return dx * dx + dy * dy;
}

void Freehand::add_control_point(float x, float y)
{
    points_.push_back(Pt{ x, y });
}

void Freehand::update(float x, float y)
{
    if (points_.empty())
    {
        points_.push_back(Pt{ x, y });
        return;
    }
    const auto& last = points_.back();
    const float th2 = min_dist_ * min_dist_;
    if (dist2(last.x, last.y, x, y) >= th2)
        points_.push_back(Pt{ x, y });
}

void Freehand::draw(const Config& config) const
{
    if (points_.size() < 2)
        return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 color = IM_COL32(
        config.line_color[0],
        config.line_color[1],
        config.line_color[2],
        config.line_color[3]);

    for (size_t i = 1; i < points_.size(); ++i)
    {
        const auto& a = points_[i - 1];
        const auto& b = points_[i];
        draw_list->AddLine(
            ImVec2(config.bias[0] + a.x, config.bias[1] + a.y),
            ImVec2(config.bias[0] + b.x, config.bias[1] + b.y),
            color,
            config.line_thickness);
    }
}
}  // namespace USTC_CG
