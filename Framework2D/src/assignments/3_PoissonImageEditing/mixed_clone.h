#pragma once

#include "poisson_clone.h"

namespace USTC_CG
{
class MixedClone : public PoissonClone
{
   public:
    MixedClone(std::shared_ptr<Image> source, std::shared_ptr<Image> mask);

   private:
    double compute_guidance(
        const Image& target,
        int src_x,
        int src_y,
        int tar_x,
        int tar_y,
        int neighbor_src_x,
        int neighbor_src_y,
        int neighbor_tar_x,
        int neighbor_tar_y,
        int channel) const override;
};
}  // namespace USTC_CG
