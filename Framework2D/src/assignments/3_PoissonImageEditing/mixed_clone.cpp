#include "mixed_clone.h"

#include <cmath>
#include <utility>

namespace USTC_CG
{
MixedClone::MixedClone(std::shared_ptr<Image> source, std::shared_ptr<Image> mask)
    : PoissonClone(std::move(source), std::move(mask))
{
    initialize();
}

double MixedClone::compute_guidance(
    const Image& target,
    int src_x,
    int src_y,
    int tar_x,
    int tar_y,
    int neighbor_src_x,
    int neighbor_src_y,
    int neighbor_tar_x,
    int neighbor_tar_y,
    int channel) const
{
    const double source_center = get_channel(source_image(), src_x, src_y, channel);
    const double source_gradient =
        source_center - get_channel(source_image(), neighbor_src_x, neighbor_src_y, channel);

    const double target_center = get_clamped_channel(target, tar_x, tar_y, channel);
    double target_gradient = source_gradient;
    if (is_inside_image(target, neighbor_tar_x, neighbor_tar_y))
    {
        target_gradient =
            target_center - get_channel(target, neighbor_tar_x, neighbor_tar_y, channel);
    }

    return std::abs(source_gradient) > std::abs(target_gradient)
               ? source_gradient
               : target_gradient;
}
}  // namespace USTC_CG
