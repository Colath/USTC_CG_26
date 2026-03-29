import torch
from forward_noising import (
    get_index_from_list, sqrt_one_minus_alphas_cumprod, betas,
    posterior_variance, sqrt_recip_alphas, T
)
import matplotlib.pyplot as plt
from dataloader import show_tensor_image
from unet import SimpleUnet

@torch.no_grad()
def sample_timestep(model, x, t):
    """
    单步去噪：从 x_t 推导 x_{t-1} [cite: 1514]
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # 公式：x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta) [cite: 881]
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        var_t = get_index_from_list(posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(var_t) * noise

@torch.no_grad()
def sample_plot_image(model, device, img_size, T):
    """
    从纯噪声开始迭代去噪并绘图 [cite: 1514]
    """
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 5))
    
    for i in range(T-1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        if i % (T // 10) == 0:
            plt.subplot(1, 11, (T-1-i)//(T//10) + 1)
            show_tensor_image(img.detach().cpu())
    plt.show()

def test_image_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUnet().to(device)
    model.load_state_dict(torch.load("ddpm_model_multi.pth", map_location=device))
    model.eval()
    sample_plot_image(model, device, 128, T)

if __name__ == "__main__":
    test_image_generation()