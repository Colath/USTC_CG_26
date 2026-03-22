#pragma once

#include <cstddef>

#include "common/image_widget.h"
#include "shapes/polygon.h"
#include "shapes/rect.h"

namespace USTC_CG
{
class SourceImageWidget : public ImageWidget
{
   public:
    enum RegionType
    {
        kDefault = 0,
        kRect = 1,
        kPolygon = 2  // 新增多边形
    };

    explicit SourceImageWidget(
        const std::string& label,
        const std::string& filename);
    virtual ~SourceImageWidget() noexcept = default;

    void draw() override;
    void enable_selecting(bool flag);
    void select_region();
    std::shared_ptr<Image> get_region_mask();
    std::shared_ptr<Image> get_data();
    ImVec2 get_position() const;
    std::size_t get_mask_version() const;

    // 设置选取工具
    void set_region_type(RegionType type)
    {
        region_type_ = type;
    }

   private:
    void mouse_click_event();
    void mouse_move_event();
    void mouse_release_event();
    ImVec2 mouse_pos_in_canvas() const;
    void update_selected_region();

    RegionType region_type_ = kRect;
    std::shared_ptr<Shape> selected_shape_;  // 改为 shared_ptr 方便多态
    std::shared_ptr<Image> selected_region_mask_;
    std::size_t mask_version_ = 0;

    ImVec2 start_, end_;
    bool flag_enable_selecting_region_ = false;
    bool draw_status_ = false;
};

}  // namespace USTC_CG