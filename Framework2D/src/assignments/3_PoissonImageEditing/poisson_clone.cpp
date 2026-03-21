#include "poisson_clone.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace USTC_CG
{
namespace
{
constexpr std::array<std::pair<int, int>, 4> kNeighbors = {
    std::pair<int, int> { -1, 0 },
    std::pair<int, int> { 1, 0 },
    std::pair<int, int> { 0, -1 },
    std::pair<int, int> { 0, 1 }
};
}

PoissonClone::PoissonClone(
    std::shared_ptr<Image> source,
    std::shared_ptr<Image> mask)
    : source_(std::move(source)),
      mask_(std::move(mask))
{
    if (source_ == nullptr || mask_ == nullptr)
        return;

    width_ = mask_->width();
    height_ = mask_->height();
    if (width_ <= 0 || height_ <= 0 || source_->width() != width_ ||
        source_->height() != height_)
    {
        width_ = 0;
        height_ = 0;
    }
}

bool PoissonClone::is_valid() const
{
    return valid_;
}

void PoissonClone::apply(
    const Image& target,
    Image& output,
    int offset_x,
    int offset_y) const
{
    if (!valid_)
        return;

    constexpr int kColorChannels = 3;
    std::vector<Eigen::VectorXd> solutions;
    solutions.reserve(kColorChannels);
    for (int channel = 0; channel < kColorChannels; ++channel)
    {
        Eigen::VectorXd rhs;
        build_rhs(target, offset_x, offset_y, channel, rhs);
        Eigen::VectorXd solution = solver_.solve(rhs);
        if (solver_.info() != Eigen::Success)
            return;
        solutions.push_back(std::move(solution));
    }

    scatter_solution(output, solutions, offset_x, offset_y);
}

void PoissonClone::initialize()
{
    if (width_ <= 0 || height_ <= 0)
        return;

    initialize_region();
    build_system();
}

const Image& PoissonClone::source_image() const
{
    return *source_;
}

double PoissonClone::get_channel(const Image& image, int x, int y, int channel)
{
    return static_cast<double>(image.data()[pixel_offset(image, x, y, channel)]);
}

double PoissonClone::get_clamped_channel(
    const Image& image,
    int x,
    int y,
    int channel)
{
    const int clamped_x = std::clamp(x, 0, image.width() - 1);
    const int clamped_y = std::clamp(y, 0, image.height() - 1);
    return get_channel(image, clamped_x, clamped_y, channel);
}

bool PoissonClone::is_inside_region(int x, int y) const
{
    return 0 <= x && x < width_ && 0 <= y && y < height_ &&
           index_map_[y * width_ + x] >= 0;
}

bool PoissonClone::is_inside_image(const Image& image, int x, int y)
{
    return 0 <= x && x < image.width() && 0 <= y && y < image.height();
}

int PoissonClone::index_of(int x, int y) const
{
    return index_map_[y * width_ + x];
}

void PoissonClone::initialize_region()
{
    index_map_.assign(width_ * height_, -1);
    const unsigned char* mask_data = mask_->data();
    const int mask_channels = mask_->channels();

    for (int y = 0; y < height_; ++y)
    {
        for (int x = 0; x < width_; ++x)
        {
            if (mask_data[(y * width_ + x) * mask_channels] == 0)
                continue;

            index_map_[y * width_ + x] = static_cast<int>(region_pixels_.size());
            region_pixels_.push_back({ x, y });
        }
    }
}

void PoissonClone::build_system()
{
    const int unknown_count = static_cast<int>(region_pixels_.size());
    if (unknown_count == 0)
        return;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(unknown_count) * 5);
    for (int row = 0; row < unknown_count; ++row)
    {
        const PixelCoord& pixel = region_pixels_[row];
        double diagonal = 0.0;
        for (const auto& [dx, dy] : kNeighbors)
        {
            const int nx = pixel.x + dx;
            const int ny = pixel.y + dy;
            if (!is_inside_image(*source_, nx, ny))
                continue;

            diagonal += 1.0;
            if (!is_inside_region(nx, ny))
                continue;
            triplets.emplace_back(row, index_of(nx, ny), -1.0);
        }

        if (diagonal <= 0.0)
            diagonal = 1.0;
        triplets.emplace_back(row, row, diagonal);
    }

    system_matrix_.resize(unknown_count, unknown_count);
    system_matrix_.setFromTriplets(triplets.begin(), triplets.end());
    solver_.compute(system_matrix_);
    valid_ = solver_.info() == Eigen::Success;
}

void PoissonClone::build_rhs(
    const Image& target,
    int offset_x,
    int offset_y,
    int channel,
    Eigen::VectorXd& rhs) const
{
    const int unknown_count = static_cast<int>(region_pixels_.size());
    rhs.resize(unknown_count);

    for (int row = 0; row < unknown_count; ++row)
    {
        const PixelCoord& pixel = region_pixels_[row];
        const int tar_x = pixel.x + offset_x;
        const int tar_y = pixel.y + offset_y;

        double value = 0.0;
        for (const auto& [dx, dy] : kNeighbors)
        {
            const int nx = pixel.x + dx;
            const int ny = pixel.y + dy;
            if (!is_inside_image(*source_, nx, ny))
                continue;

            const int target_nx = tar_x + dx;
            const int target_ny = tar_y + dy;
            value += compute_guidance(
                target,
                pixel.x,
                pixel.y,
                tar_x,
                tar_y,
                nx,
                ny,
                target_nx,
                target_ny,
                channel);

            if (is_inside_region(nx, ny))
                continue;

            if (is_inside_image(target, target_nx, target_ny))
            {
                value += get_channel(target, target_nx, target_ny, channel);
            }
        }

        rhs[row] = value;
    }
}

void PoissonClone::scatter_solution(
    Image& output,
    const std::vector<Eigen::VectorXd>& solutions,
    int offset_x,
    int offset_y) const
{
    constexpr int kColorChannels = 3;
    for (int row = 0; row < static_cast<int>(region_pixels_.size()); ++row)
    {
        const PixelCoord& pixel = region_pixels_[row];
        const int tar_x = pixel.x + offset_x;
        const int tar_y = pixel.y + offset_y;
        if (!is_inside_image(output, tar_x, tar_y))
            continue;

        for (int channel = 0;
             channel < kColorChannels && channel < output.channels();
             ++channel)
        {
            output.data()[pixel_offset(output, tar_x, tar_y, channel)] =
                clamp_to_uchar(solutions[channel][row]);
        }
    }
}

unsigned char PoissonClone::clamp_to_uchar(double value)
{
    const int rounded = static_cast<int>(std::lround(value));
    return static_cast<unsigned char>(std::clamp(rounded, 0, 255));
}

int PoissonClone::pixel_offset(const Image& image, int x, int y, int channel)
{
    return ((y * image.width()) + x) * image.channels() + channel;
}
}  // namespace USTC_CG
