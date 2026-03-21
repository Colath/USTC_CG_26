#include "seamless_clone.h"

#include <utility>

namespace USTC_CG
{
SeamlessClone::SeamlessClone(
    std::shared_ptr<Image> source,
    std::shared_ptr<Image> mask)
    : PoissonClone(std::move(source), std::move(mask))
{
    initialize();
}

double SeamlessClone::compute_guidance(
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
    (void)target;
    (void)tar_x;
    (void)tar_y;
    (void)neighbor_tar_x;
    (void)neighbor_tar_y;
    const double source_center =
        get_clamped_channel(source_image(), src_x, src_y, channel);
    return source_center - get_clamped_channel(
                               source_image(),
                               neighbor_src_x,
                               neighbor_src_y,
                               channel);
}
}  // namespace USTC_CG
