#include "polygon.h"

#include <imgui.h>

#include <algorithm>

namespace USTC_CG
{
void Polygon::draw(const Config& config) const
{
    if (vertices_.empty())
        return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImU32 color = IM_COL32(
        config.line_color[0],
        config.line_color[1],
        config.line_color[2],
        config.line_color[3]);

    for (size_t i = 0; i < vertices_.size() - 1; ++i)
    {
        draw_list->AddLine(
            ImVec2(
                config.bias[0] + vertices_[i].first,
                config.bias[1] + vertices_[i].second),
            ImVec2(
                config.bias[0] + vertices_[i + 1].first,
                config.bias[1] + vertices_[i + 1].second),
            color,
            config.line_thickness);
    }

    if (!is_finished_)
    {
        draw_list->AddLine(
            ImVec2(
                config.bias[0] + vertices_.back().first,
                config.bias[1] + vertices_.back().second),
            ImVec2(config.bias[0] + current_x_, config.bias[1] + current_y_),
            color,
            config.line_thickness);
    }
    else  
    {
        draw_list->AddLine(
            ImVec2(
                config.bias[0] + vertices_.back().first,
                config.bias[1] + vertices_.back().second),
            ImVec2(
                config.bias[0] + vertices_.front().first,
                config.bias[1] + vertices_.front().second),
            color,
            config.line_thickness);
    }
}

void Polygon::update(float x, float y)
{
    if (!is_finished_)
    {
        current_x_ = x;
        current_y_ = y;
    }
}

void Polygon::add_control_point(float x, float y)
{
    if (!is_finished_)
    {
        vertices_.push_back({ x, y });
    }
}

void Polygon::finish_drawing()
{
    if (vertices_.size() >= 3)
    {
        is_finished_ = true;
    }
}

std::vector<std::pair<int, int>> Polygon::get_interior_pixels() const
{
    std::vector<std::pair<int, int>> interior_pixels;
    if (vertices_.size() < 3)
        return interior_pixels;

    int min_y = 1e9, max_y = -1e9;
    for (const auto& v : vertices_)
    {
        min_y = std::min(min_y, static_cast<int>(v.second));
        max_y = std::max(max_y, static_cast<int>(v.second));
    }

    for (int y = min_y; y <= max_y; ++y)
    {
        std::vector<int> intersections;
        for (size_t i = 0; i < vertices_.size(); ++i)
        {
            size_t j = (i + 1) % vertices_.size();
            float x1 = vertices_[i].first, y1 = vertices_[i].second;
            float x2 = vertices_[j].first, y2 = vertices_[j].second;

            if (y1 > y2)
            {
                std::swap(x1, x2);
                std::swap(y1, y2);
            }

            if (y >= y1 && y < y2)
            {
                float x = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                intersections.push_back(static_cast<int>(x));
            }
        }

        std::sort(intersections.begin(), intersections.end());

        for (size_t i = 0; i + 1 < intersections.size(); i += 2)
        {
            for (int x = intersections[i]; x <= intersections[i + 1]; ++x)
            {
                interior_pixels.push_back({ x, y });
            }
        }
    }
    return interior_pixels;
}
}  // namespace USTC_CG