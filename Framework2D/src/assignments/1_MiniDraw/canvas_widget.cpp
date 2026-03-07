#include "canvas_widget.h"

#include <cmath>
#include <iostream>
#include <memory>  

#include "imgui.h"
#include "shapes/ellipse.h"
#include "shapes/freehand.h"
#include "shapes/line.h"
#include "shapes/polygon.h"
#include "shapes/rect.h"

namespace USTC_CG
{
void Canvas::draw()
{
    draw_background();

    // Polygon: Enter to finish and get the polygon closed, Esc to cancel the current polygon
    if (is_hovered_ && shape_type_ == kPolygon && draw_status_ &&
        current_shape_)
    {
        if (ImGui::IsKeyPressed(ImGuiKey_Enter))
        {
            auto poly = std::dynamic_pointer_cast<Polygon>(current_shape_);
            if (poly && poly->num_points() >= 3)
            {
                poly->finish();
                shape_list_.push_back(current_shape_);
            }
            current_shape_.reset();
            draw_status_ = false;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape))
        {
            current_shape_.reset();
            draw_status_ = false;
        }
    }

    // Pure Click-Click interaction:
    //   - First click starts a shape
    //   - Mouse move updates preview
    //   - Second click finishes (except Polygon which uses Enter)
    if (is_hovered_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        mouse_click_event();

    mouse_move_event();

    draw_shapes();
}

void Canvas::set_attributes(const ImVec2& min, const ImVec2& size)
{
    canvas_min_ = min;
    canvas_size_ = size;
    canvas_minimal_size_ = size;
    canvas_max_ =
        ImVec2(canvas_min_.x + canvas_size_.x, canvas_min_.y + canvas_size_.y);
}

void Canvas::show_background(bool flag)
{
    show_background_ = flag;
}

void Canvas::set_default()
{
    draw_status_ = false;
    current_shape_.reset();
    shape_type_ = kDefault;
}

void Canvas::set_line()
{
    draw_status_ = false;
    current_shape_.reset();
    shape_type_ = kLine;
}

void Canvas::set_rect()
{
    draw_status_ = false;
    current_shape_.reset();
    shape_type_ = kRect;
}

void Canvas::set_ellipse()
{
    draw_status_ = false;
    current_shape_.reset();
    shape_type_ = kEllipse;
}

void Canvas::set_polygon()
{
    draw_status_ = false;
    current_shape_.reset();
    shape_type_ = kPolygon;
}

void Canvas::set_freehand()
{
    draw_status_ = false;
    current_shape_.reset();
    shape_type_ = kFreehand;
}

void Canvas::clear_shape_list()
{
    shape_list_.clear();
}

void Canvas::cancel_current()
{
    current_shape_.reset();
    draw_status_ = false;
}

void Canvas::undo()
{
    cancel_current();
    if (!shape_list_.empty())
        shape_list_.pop_back();
}

void Canvas::clear_all()
{
    cancel_current();
    shape_list_.clear();
}

void Canvas::draw_background()
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    if (show_background_)
    {
        // Draw background recrangle
        draw_list->AddRectFilled(canvas_min_, canvas_max_, background_color_);
        // Draw background border
        draw_list->AddRect(canvas_min_, canvas_max_, border_color_);
    }
    /// Invisible button over the canvas to capture mouse interactions.
    ImGui::SetCursorScreenPos(canvas_min_);
    ImGui::InvisibleButton(
        label_.c_str(), canvas_size_, ImGuiButtonFlags_MouseButtonLeft);
    // Record the current status of the invisible button
    is_hovered_ = ImGui::IsItemHovered();
    is_active_ = ImGui::IsItemActive();
}

void Canvas::draw_shapes()
{
    Shape::Config s = { .bias = { canvas_min_.x, canvas_min_.y } };
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // ClipRect can hide the drawing content outside of the rectangular area
    draw_list->PushClipRect(canvas_min_, canvas_max_, true);
    for (const auto& shape : shape_list_)
    {
        shape->draw(s);
    }
    if (draw_status_ && current_shape_)
    {
        current_shape_->draw(s);
    }
    draw_list->PopClipRect();
}

void Canvas::mouse_click_event()
{
    if (shape_type_ == kPolygon)
    {
        const ImVec2 p = mouse_pos_in_canvas();
        if (!draw_status_ || !current_shape_)
        {
            draw_status_ = true;
            current_shape_ = std::make_shared<Polygon>();
        }
        current_shape_->add_control_point(p.x, p.y);
        return;
    }

    if (shape_type_ == kFreehand)
    {
        const ImVec2 p = mouse_pos_in_canvas();
        if (!draw_status_ || !current_shape_)
        {
            draw_status_ = true;
            current_shape_ = std::make_shared<Freehand>();
            current_shape_->add_control_point(p.x, p.y);
        }
        else
        {
            shape_list_.push_back(current_shape_);
            current_shape_.reset();
            draw_status_ = false;
        }
        return;
    }

    if (!draw_status_)
    {
        draw_status_ = true;
        start_point_ = end_point_ = mouse_pos_in_canvas();
        switch (shape_type_)
        {
            case kLine:
                current_shape_ = std::make_shared<Line>(
                    start_point_.x, start_point_.y, end_point_.x, end_point_.y);
                break;
            case kRect:
                current_shape_ = std::make_shared<Rect>(
                    start_point_.x, start_point_.y, end_point_.x, end_point_.y);
                break;
            case kEllipse:
                current_shape_ = std::make_shared<Ellipse>(
                    start_point_.x, start_point_.y, end_point_.x, end_point_.y);
                break;
            default: break;
        }

        // If the current tool does not create a shape (e.g. default), stop.
        if (!current_shape_)
            draw_status_ = false;
    }
    else
    {
        draw_status_ = false;
        if (current_shape_)
        {
            shape_list_.push_back(current_shape_);
            current_shape_.reset();
        }
    }
}

void Canvas::mouse_move_event()
{
    if (!draw_status_ || !current_shape_)
        return;

    end_point_ = mouse_pos_in_canvas();

    current_shape_->update(end_point_.x, end_point_.y);
}

ImVec2 Canvas::mouse_pos_in_canvas() const
{
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse_pos_in_canvas(
        io.MousePos.x - canvas_min_.x, io.MousePos.y - canvas_min_.y);
    return mouse_pos_in_canvas;
}
}  // namespace USTC_CG
