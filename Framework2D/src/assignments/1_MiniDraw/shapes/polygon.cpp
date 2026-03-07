#include "polygon.h"

#include <imgui.h>

namespace USTC_CG
{
void Polygon::add_control_point(float x, float y)
{
    points_.push_back(Pt{ x, y });
}

void Polygon::update(float x, float y)
{
    // only show rubber band while drawing (not finished)
    if (!finished_)
    {
        has_preview_ = true;
        preview_ = Pt{ x, y };
    }
}

void Polygon::draw(const Config& config) const
{
    if (points_.empty())
        return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImU32 color = IM_COL32(
        config.line_color[0],
        config.line_color[1],
        config.line_color[2],
        config.line_color[3]);

    // draw edges between fixed points: p[i-1] -> p[i]
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

    // rubber band preview: last point -> mouse 
    if (!finished_ && has_preview_ && !points_.empty())
    {
        const auto& last = points_.back();
        draw_list->AddLine(
            ImVec2(config.bias[0] + last.x, config.bias[1] + last.y),
            ImVec2(config.bias[0] + preview_.x, config.bias[1] + preview_.y),
            color,
            config.line_thickness);
    }

    // close polygon: last -> first 
    if (finished_ && points_.size() >= 3)
    {
        const auto& first = points_.front();
        const auto& last = points_.back();
        draw_list->AddLine(
            ImVec2(config.bias[0] + last.x, config.bias[1] + last.y),
            ImVec2(config.bias[0] + first.x, config.bias[1] + first.y),
            color,
            config.line_thickness);
    }
}
}  // namespace USTC_CG
