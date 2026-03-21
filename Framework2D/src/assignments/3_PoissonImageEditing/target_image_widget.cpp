#include "target_image_widget.h"

#include <algorithm>
#include <cmath>

#include "mixed_clone.h"
#include "seamless_clone.h"

namespace USTC_CG
{
using uchar = unsigned char;

namespace
{
struct MaskBounds
{
    int min_x = 0;
    int min_y = 0;
    int max_x = 0;
    int max_y = 0;
    bool valid = false;
};

MaskBounds compute_mask_bounds(const Image& mask)
{
    MaskBounds bounds;
    const int channels = mask.channels();
    for (int y = 0; y < mask.height(); ++y)
    {
        for (int x = 0; x < mask.width(); ++x)
        {
            const int index = (y * mask.width() + x) * channels;
            if (mask.data()[index] == 0)
                continue;

            if (!bounds.valid)
            {
                bounds.min_x = bounds.max_x = x;
                bounds.min_y = bounds.max_y = y;
                bounds.valid = true;
                continue;
            }

            bounds.min_x = std::min(bounds.min_x, x);
            bounds.min_y = std::min(bounds.min_y, y);
            bounds.max_x = std::max(bounds.max_x, x);
            bounds.max_y = std::max(bounds.max_y, y);
        }
    }

    return bounds;
}

ImVec2 clamp_clone_anchor(
    const ImVec2& anchor,
    const Image& mask,
    const ImVec2& source_position,
    int target_width,
    int target_height)
{
    const MaskBounds bounds = compute_mask_bounds(mask);
    if (!bounds.valid)
        return anchor;

    int margin = 1;
    float min_x =
        source_position.x + static_cast<float>(margin - bounds.min_x);
    float min_y =
        source_position.y + static_cast<float>(margin - bounds.min_y);
    float max_x = source_position.x +
                  static_cast<float>(target_width - 1 - margin - bounds.max_x);
    float max_y = source_position.y + static_cast<float>(
                                      target_height - 1 - margin - bounds.max_y);

    if (min_x > max_x || min_y > max_y)
    {
        margin = 0;
        min_x = source_position.x + static_cast<float>(margin - bounds.min_x);
        min_y = source_position.y + static_cast<float>(margin - bounds.min_y);
        max_x = source_position.x + static_cast<float>(
                                      target_width - 1 - margin - bounds.max_x);
        max_y = source_position.y + static_cast<float>(
                                      target_height - 1 - margin - bounds.max_y);
    }

    if (min_x > max_x || min_y > max_y)
        return anchor;

    return ImVec2(
        std::clamp(anchor.x, min_x, max_x),
        std::clamp(anchor.y, min_y, max_y));
}
}  // namespace

TargetImageWidget::TargetImageWidget(
    const std::string& label,
    const std::string& filename)
    : ImageWidget(label, filename)
{
    if (data_)
        back_up_ = std::make_shared<Image>(*data_);
}

void TargetImageWidget::draw()
{
    // Draw the image
    ImageWidget::draw();
    // Invisible button for interactions
    ImGui::SetCursorScreenPos(position_);
    ImGui::InvisibleButton(
        label_.c_str(),
        ImVec2(
            static_cast<float>(image_width_),
            static_cast<float>(image_height_)),
        ImGuiButtonFlags_MouseButtonLeft);
    bool is_hovered_ = ImGui::IsItemHovered();
    // When the mouse is clicked or moving, we would adapt clone function to
    // copy the selected region to the target.

    if (is_hovered_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        mouse_click_event();
    }
    mouse_move_event();
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
    {
        mouse_release_event();
    }
}

void TargetImageWidget::set_source(std::shared_ptr<SourceImageWidget> source)
{
    source_image_ = source;
    mixed_clone_.reset();
    seamless_clone_.reset();
    mixed_mask_version_ = 0;
    seamless_mask_version_ = 0;
}

void TargetImageWidget::set_realtime(bool flag)
{
    flag_realtime_updating = flag;
}

void TargetImageWidget::restore()
{
    *data_ = *back_up_;
    update();
}

void TargetImageWidget::set_paste()
{
    clone_type_ = kPaste;
}

void TargetImageWidget::set_seamless()
{
    clone_type_ = kSeamless;
}

void TargetImageWidget::set_mixed()
{
    clone_type_ = kMixed;
}

bool TargetImageWidget::prepare_mixed_clone()
{
    if (source_image_ == nullptr)
        return false;

    std::shared_ptr<Image> source = source_image_->get_data();
    std::shared_ptr<Image> mask = source_image_->get_region_mask();
    if (source == nullptr || mask == nullptr)
        return false;

    const ImVec2 source_position = source_image_->get_position();
    const std::size_t mask_version = source_image_->get_mask_version();
    if (
        mixed_clone_ == nullptr || mixed_mask_version_ != mask_version ||
        mixed_source_position_.x != source_position.x ||
        mixed_source_position_.y != source_position.y)
    {
        mixed_clone_ = std::make_unique<MixedClone>(source, mask);
        mixed_mask_version_ = mask_version;
        mixed_source_position_ = source_position;
    }

    return mixed_clone_ != nullptr && mixed_clone_->is_valid();
}

bool TargetImageWidget::prepare_seamless_clone()
{
    if (source_image_ == nullptr)
        return false;

    std::shared_ptr<Image> source = source_image_->get_data();
    std::shared_ptr<Image> mask = source_image_->get_region_mask();
    if (source == nullptr || mask == nullptr)
        return false;

    const ImVec2 source_position = source_image_->get_position();
    const std::size_t mask_version = source_image_->get_mask_version();
    if (
        seamless_clone_ == nullptr || seamless_mask_version_ != mask_version ||
        seamless_source_position_.x != source_position.x ||
        seamless_source_position_.y != source_position.y)
    {
        seamless_clone_ = std::make_unique<SeamlessClone>(source, mask);
        seamless_mask_version_ = mask_version;
        seamless_source_position_ = source_position;
    }

    return seamless_clone_ != nullptr && seamless_clone_->is_valid();
}

void TargetImageWidget::clone()
{
    // The implementation of different types of cloning
    // HW3_TODO: 
    // 1. In this function, you should at least implement the "seamless"
    // cloning labeled by `clone_type_ ==kSeamless`.
    //
    // 2. It is required to improve the efficiency of your seamless cloning to
    // achieve real-time editing. (Use decomposition of sparse matrix before
    // solve the linear system). The real-time updating (update when the mouse
    // is moving) is only available when the checkerboard is selected. 
    if (data_ == nullptr || source_image_ == nullptr ||
        source_image_->get_region_mask() == nullptr)
        return;
    // The selected region in the source image, this would be a binary mask.
    // The **size** of the mask should be the same as the source image.
    // The **value** of the mask should be 0 or 255: 0 for the background and
    // 255 for the selected region.
    std::shared_ptr<Image> mask = source_image_->get_region_mask();
    mouse_position_ = clamp_clone_anchor(
        mouse_position_,
        *mask,
        source_image_->get_position(),
        image_width_,
        image_height_);

    switch (clone_type_)
    {
        case USTC_CG::TargetImageWidget::kDefault: break;
        case USTC_CG::TargetImageWidget::kPaste:
        {
            restore();

            for (int x = 0; x < mask->width(); ++x)
            {
                for (int y = 0; y < mask->height(); ++y)
                {
                    int tar_x =
                        static_cast<int>(mouse_position_.x) + x -
                        static_cast<int>(source_image_->get_position().x);
                    int tar_y =
                        static_cast<int>(mouse_position_.y) + y -
                        static_cast<int>(source_image_->get_position().y);
                    if (0 <= tar_x && tar_x < image_width_ && 0 <= tar_y &&
                        tar_y < image_height_ && mask->get_pixel(x, y)[0] > 0)
                    {
                        data_->set_pixel(
                            tar_x,
                            tar_y,
                            source_image_->get_data()->get_pixel(x, y));
                    }
                }
            }
            break;
        }
        case USTC_CG::TargetImageWidget::kSeamless:
        {
            restore();
            if (!prepare_seamless_clone() || back_up_ == nullptr)
                break;

            const int offset_x = static_cast<int>(std::lround(
                mouse_position_.x - seamless_source_position_.x));
            const int offset_y = static_cast<int>(std::lround(
                mouse_position_.y - seamless_source_position_.y));
            seamless_clone_->apply(*back_up_, *data_, offset_x, offset_y);

            break;
        }
        case USTC_CG::TargetImageWidget::kMixed:
        {
            restore();
            if (!prepare_mixed_clone() || back_up_ == nullptr)
                break;

            const int offset_x = static_cast<int>(std::lround(
                mouse_position_.x - mixed_source_position_.x));
            const int offset_y = static_cast<int>(std::lround(
                mouse_position_.y - mixed_source_position_.y));
            mixed_clone_->apply(*back_up_, *data_, offset_x, offset_y);

            break;
        }
        default: break;
    }

    update();
}

void TargetImageWidget::mouse_click_event()
{
    edit_status_ = true;
    mouse_position_ = mouse_pos_in_canvas();
    clone();
}

void TargetImageWidget::mouse_move_event()
{
    if (edit_status_)
    {
        mouse_position_ = mouse_pos_in_canvas();
        if (flag_realtime_updating)
            clone();
    }
}

void TargetImageWidget::mouse_release_event()
{
    if (edit_status_)
    {
        edit_status_ = false;
    }
}

ImVec2 TargetImageWidget::mouse_pos_in_canvas() const
{
    ImGuiIO& io = ImGui::GetIO();
    return ImVec2(io.MousePos.x - position_.x, io.MousePos.y - position_.y);
}
}  // namespace USTC_CG