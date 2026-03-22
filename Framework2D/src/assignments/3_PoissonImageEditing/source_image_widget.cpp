#include "source_image_widget.h"

#include <algorithm>
#include <cmath>

namespace USTC_CG
{
using uchar = unsigned char;

SourceImageWidget::SourceImageWidget(
    const std::string& label,
    const std::string& filename)
    : ImageWidget(label, filename)
{
    if (data_)
        selected_region_mask_ =
            std::make_shared<Image>(data_->width(), data_->height(), 1);
}

void SourceImageWidget::draw()
{
    ImageWidget::draw();
    if (flag_enable_selecting_region_)
        select_region();
}

void SourceImageWidget::enable_selecting(bool flag)
{
    flag_enable_selecting_region_ = flag;
}

void SourceImageWidget::select_region()
{
    ImGui::SetCursorScreenPos(position_);
    ImGui::InvisibleButton(
        label_.c_str(),
        ImVec2(
            static_cast<float>(image_width_),
            static_cast<float>(image_height_)),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);

    bool is_hovered_ = ImGui::IsItemHovered();

    // 左键点击（矩形起点，或多边形加点）
    if (is_hovered_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        mouse_click_event();

    // 鼠标移动
    mouse_move_event();

    // 矩形的松开逻辑：左键松开即完成
    if (region_type_ == kRect && !ImGui::IsMouseDown(ImGuiMouseButton_Left))
        mouse_release_event();

    // 多边形的完成逻辑：右键闭合并完成
    if (region_type_ == kPolygon && is_hovered_ &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Right))
    {
        if (draw_status_ && selected_shape_)
        {
            auto poly = std::dynamic_pointer_cast<Polygon>(selected_shape_);
            if (poly)
            {
                poly->finish_drawing();
                draw_status_ = false;
                update_selected_region();
            }
        }
    }

    if (selected_shape_)
    {
        Shape::Config s = { .bias = { position_.x, position_.y },
                            .line_color = { 255, 0, 0, 255 },
                            .line_thickness = 2.0f };
        selected_shape_->draw(s);
    }
}

std::shared_ptr<Image> SourceImageWidget::get_region_mask()
{
    return selected_region_mask_;
}
std::shared_ptr<Image> SourceImageWidget::get_data()
{
    return data_;
}
std::size_t SourceImageWidget::get_mask_version() const
{
    return mask_version_;
}

ImVec2 SourceImageWidget::get_position() const
{
    // 对多边形来说，最好返回包围盒左上角，这里用一种简单的通用近似（如果选了多边形，最好在选完后更新
    // start 和 end）
    return ImVec2(std::min(start_.x, end_.x), std::min(start_.y, end_.y));
}

void SourceImageWidget::mouse_click_event()
{
    if (region_type_ == kRect)
    {
        if (!draw_status_)
        {
            draw_status_ = true;
            start_ = end_ = mouse_pos_in_canvas();
            selected_shape_ =
                std::make_shared<Rect>(start_.x, start_.y, end_.x, end_.y);
        }
    }
    else if (region_type_ == kPolygon)
    {
        if (!draw_status_)
        {
            draw_status_ = true;
            selected_shape_ = std::make_shared<Polygon>();
            // 记录初始点作为近似偏移锚点
            start_ = end_ = mouse_pos_in_canvas();
        }
        ImVec2 pos = mouse_pos_in_canvas();
        selected_shape_->add_control_point(pos.x, pos.y);

        // 更新包围盒范围，确保 get_position() 相对正确
        start_.x = std::min(start_.x, pos.x);
        start_.y = std::min(start_.y, pos.y);
        end_.x = std::max(end_.x, pos.x);
        end_.y = std::max(end_.y, pos.y);
    }
}

void SourceImageWidget::mouse_move_event()
{
    if (draw_status_ && selected_shape_)
    {
        ImVec2 pos = mouse_pos_in_canvas();
        if (region_type_ == kRect)
            end_ = pos;
        selected_shape_->update(pos.x, pos.y);
    }
}

void SourceImageWidget::mouse_release_event()
{
    if (draw_status_ && selected_shape_)
    {
        draw_status_ = false;
        update_selected_region();
    }
}

ImVec2 SourceImageWidget::mouse_pos_in_canvas() const
{
    ImGuiIO& io = ImGui::GetIO();
    return ImVec2(
        std::clamp<float>(io.MousePos.x - position_.x, 0, (float)image_width_),
        std::clamp<float>(
            io.MousePos.y - position_.y, 0, (float)image_height_));
}

void SourceImageWidget::update_selected_region()
{
    if (selected_shape_ == nullptr)
        return;

    std::vector<std::pair<int, int>> interior_pixels;

    if (auto rect = std::dynamic_pointer_cast<Rect>(selected_shape_))
        interior_pixels = rect->get_interior_pixels();
    else if (auto poly = std::dynamic_pointer_cast<Polygon>(selected_shape_))
        interior_pixels = poly->get_interior_pixels();

    for (int i = 0; i < selected_region_mask_->width(); ++i)
        for (int j = 0; j < selected_region_mask_->height(); ++j)
            selected_region_mask_->set_pixel(i, j, { 0 });

    for (const auto& pixel : interior_pixels)
    {
        int x = pixel.first;
        int y = pixel.second;
        if (x < 0 || x >= selected_region_mask_->width() || y < 0 ||
            y >= selected_region_mask_->height())
            continue;
        selected_region_mask_->set_pixel(x, y, { 255 });
    }

    mask_version_++;  // 通知 Target 重新计算系数矩阵
}
}  // namespace USTC_CG