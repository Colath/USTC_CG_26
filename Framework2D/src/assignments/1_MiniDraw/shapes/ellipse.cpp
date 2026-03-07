#include "ellipse.h"

#include <imgui.h>

namespace USTC_CG
{

void Ellipse::draw(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float center_x = (start_point_x_ + end_point_x_) * 0.5f;
    float center_y = (start_point_y_ + end_point_y_) * 0.5f;

    float radius_x = (end_point_x_ - start_point_x_) * 0.5f;
    float radius_y = (end_point_y_ - start_point_y_) * 0.5f;

    if (radius_x < 0)
        radius_x = -radius_x;
    if (radius_y < 0)
        radius_y = -radius_y;

    draw_list->AddEllipse(
        ImVec2(config.bias[0] + center_x, config.bias[1] + center_y),
        ImVec2(radius_x, radius_y),
        IM_COL32(
            config.line_color[0],
            config.line_color[1],
            config.line_color[2],
            config.line_color[3]),
        0.0f,  
        0,     
        config.line_thickness);
}

void Ellipse::update(float x, float y)
{
    end_point_x_ = x;
    end_point_y_ = y;
}

}  // namespace USTC_CG