import torch
import matplotlib.pyplot as plt
from forward_noising import (
    get_index_from_list, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, betas, T
)
from unet import SimpleUnet
from sampling import sample_timestep
from dataloader import load_transformed_dataset, show_tensor_image

@torch.no_grad()
def repaint_timestep(model, x_t, x_0_gt, mask, t, device):
    """
    单步 RePaint 拼接：结合真实图像的前向加噪和模型生成的逆向去噪
    """
    # 1. 针对未知区域：用模型从 x_t 预测 x_{t-1}
    x_t_minus_1_unknown = sample_timestep(model, x_t, t)
    
    # 2. 针对已知区域：将真实的 x_0 前向加噪到 t-1 步
    t_minus_1 = t - 1
    if t_minus_1[0] >= 0:
        sqrt_alphas_cumprod_t_m1 = get_index_from_list(sqrt_alphas_cumprod, t_minus_1, x_0_gt.shape)
        sqrt_one_minus_alphas_cumprod_t_m1 = get_index_from_list(sqrt_one_minus_alphas_cumprod, t_minus_1, x_0_gt.shape)
        noise = torch.randn_like(x_0_gt)
        x_t_minus_1_known = sqrt_alphas_cumprod_t_m1 * x_0_gt + sqrt_one_minus_alphas_cumprod_t_m1 * noise
    else:
        # 最后一步，已知区域直接替换为完美无噪的真实原图
        x_t_minus_1_known = x_0_gt

    # 3. 按照 Mask 掩码将已知和未知区域拼合
    # mask 中 1 表示已知区域，0 表示需要模型生成的区域
    x_t_minus_1 = mask * x_t_minus_1_known + (1. - mask) * x_t_minus_1_unknown
    
    return x_t_minus_1

@torch.no_grad()
def run_repaint_generation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUnet().to(device)
    IMG_SIZE = 128
    
    # 加载已训练好的权重
    try:
        model.load_state_dict(torch.load("ddpm_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        print("错误：未找到 ddpm_model.pth，请先运行 training_model.py 训练模型。")
        return
        
    # 1. 取一张真实图片作为 Ground Truth
    dataloader = load_transformed_dataset(img_size=IMG_SIZE, batch_size=1)
    x_0_gt, _ = next(iter(dataloader))
    x_0_gt = x_0_gt.to(device)
    
    # 2. 构造掩码 Mask (这里演示：挖掉画面中心 64x64 的正方形区域)
    mask = torch.ones_like(x_0_gt).to(device)
    c = IMG_SIZE // 2
    mask[:, :, c-32:c+32, c-32:c+32] = 0.
    
    # 3. 初始化 x_{T-1}：
    # 未知区域使用纯噪声；已知区域使用 q(x_{T-1} | x_0) 与论文设置一致
    t_init = torch.full((1,), T - 1, device=device, dtype=torch.long)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t_init, x_0_gt.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t_init, x_0_gt.shape)
    known_noisy = sqrt_alphas_cumprod_t * x_0_gt + sqrt_one_minus_alphas_cumprod_t * torch.randn_like(x_0_gt)
    unknown_noisy = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
    x_t = mask * known_noisy + (1. - mask) * unknown_noisy
    
    # RePaint 论文中的 Resampling 次数 (如果显存和时间允许，设为 2~5 能让接缝更自然)
    # 当 u > 1 时，每走完一步都会再加噪退回一步重新生成，以协调边缘信息
    jump_n_sample = 3
    
    print("开始 RePaint 图像填充...")
    for i in range(T-1, -1, -1):
        for u in range(jump_n_sample):
            t_tensor = torch.full((1,), i, device=device, dtype=torch.long)
            
            # 第一步：计算降噪融合后的 x_{t-1}
            x_t_minus_1 = repaint_timestep(model, x_t, x_0_gt, mask, t_tensor, device)
            
            # 第二步：如果开启了 Resampling 且不在最后一步，则重新加噪回 x_t 再次洗练
            if u < jump_n_sample - 1 and i > 0:
                beta_t = get_index_from_list(betas, t_tensor, x_t_minus_1.shape)
                noise = torch.randn_like(x_t_minus_1)
                # 使用单步前向加噪公式：x_t = sqrt(1-beta)*x_{t-1} + sqrt(beta)*noise
                x_t = torch.sqrt(1. - beta_t) * x_t_minus_1 + torch.sqrt(beta_t) * noise
            else:
                x_t = x_t_minus_1
                
        if i % (T // 10) == 0:
            print(f"剩余步数: {i}")

    # 4. 可视化结果对比
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    show_tensor_image(x_0_gt.cpu())
    
    plt.subplot(1, 3, 2)
    plt.title("Masked Input (Missing Center)")
    show_tensor_image((x_0_gt * mask).cpu())
    
    plt.subplot(1, 3, 3)
    plt.title("RePaint Output")
    show_tensor_image(x_t.cpu())
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_repaint_generation()