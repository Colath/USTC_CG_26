import json
import torch
import matplotlib.pyplot as plt

from cond_unet import ConditionalUnet
from dataloader import show_tensor_image
from forward_noising import (
    T,
    betas,
    posterior_variance,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    get_index_from_list,
)


@torch.no_grad()
def sample_timestep_cond(model, x, t, class_ids):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_omab_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_a_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_a_t * (x - betas_t * model(x, t, class_ids) / sqrt_omab_t)
    if t[0] == 0:
        return model_mean
    noise = torch.randn_like(x)
    var_t = get_index_from_list(posterior_variance, t, x.shape)
    return model_mean + torch.sqrt(var_t) * noise


@torch.no_grad()
def sample_from_prompt(model, device, class_idx, img_size=128):
    x = torch.randn((1, 3, img_size, img_size), device=device)
    class_ids = torch.tensor([class_idx], device=device, dtype=torch.long)

    plt.figure(figsize=(15, 5))
    for i in range(T - 1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        x = sample_timestep_cond(model, x, t, class_ids)
        if i % (T // 10) == 0:
            plt.subplot(1, 11, (T - 1 - i) // (T // 10) + 1)
            show_tensor_image(x.detach().cpu())
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("text2img_classes.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]
    class_to_idx = meta["class_to_idx"]

    print("Available prompts:", classes)
    prompt = input("Input prompt (class name): ").strip()
    if prompt not in class_to_idx:
        raise ValueError(f"Prompt '{prompt}' not found in class names: {classes}")

    model = ConditionalUnet(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("ddpm_model_text2img.pth", map_location=device))
    model.eval()
    sample_from_prompt(model, device, class_to_idx[prompt], img_size=128)
