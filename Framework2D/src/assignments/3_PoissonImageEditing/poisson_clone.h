#pragma once

#include <memory>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include "common/image.h"

namespace USTC_CG
{
class PoissonClone
{
   public:
    PoissonClone(std::shared_ptr<Image> source, std::shared_ptr<Image> mask);
    virtual ~PoissonClone() = default;

    bool is_valid() const;
    void apply(const Image& target, Image& output, int offset_x, int offset_y)
        const;

   protected:
    struct PixelCoord
    {
        int x = 0;
        int y = 0;
    };

    void initialize();

    static double get_channel(const Image& image, int x, int y, int channel);
    static double get_clamped_channel(
        const Image& image,
        int x,
        int y,
        int channel);

    const Image& source_image() const;

    virtual double compute_guidance(
        const Image& target,
        int src_x,
        int src_y,
        int tar_x,
        int tar_y,
        int neighbor_src_x,
        int neighbor_src_y,
        int neighbor_tar_x,
        int neighbor_tar_y,
        int channel) const = 0;

    bool is_inside_region(int x, int y) const;
    static bool is_inside_image(const Image& image, int x, int y);
    int index_of(int x, int y) const;

    std::shared_ptr<Image> source_;
    std::shared_ptr<Image> mask_;
    int width_ = 0;
    int height_ = 0;

   private:
    void initialize_region();
    void build_system();
    void build_rhs(
        const Image& target,
        int offset_x,
        int offset_y,
        int channel,
        Eigen::VectorXd& rhs) const;
    void scatter_solution(
        Image& output,
        const std::vector<Eigen::VectorXd>& solutions,
        int offset_x,
        int offset_y) const;
    static unsigned char clamp_to_uchar(double value);
    static int pixel_offset(const Image& image, int x, int y, int channel);

    std::vector<int> index_map_;
    std::vector<PixelCoord> region_pixels_;

    Eigen::SparseMatrix<double> system_matrix_;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_;
    bool valid_ = false;
};
}  // namespace USTC_CG
