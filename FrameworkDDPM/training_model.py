from forward_noising import forward_diffusion_sample, T
from unet import SimpleUnet
from dataloader import load_transformed_dataset
import torch.nn.functional as F
import torch
from torch.optim import Adam
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def get_loss(model, x_0, t, device):
    """
    计算训练 Loss：预测噪声与真实噪声的 MSE [cite: 897, 1515]
    """
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUnet().to(device) 
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    IMG_SIZE = 128
    DATASET_ROOT = "./datasets-1"
    BATCH_SIZE = 8 # 根据显存调整
    epochs = 125
    dataloader = load_transformed_dataset(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        dataset_root=DATASET_ROOT,
        use_test_split=False,
        augment=False,
        repeat=8,
    )
    model.train()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch, _ in pbar:
            optimizer.zero_grad()
            
            # 随机采样时间步 [cite: 1517]
            t = torch.randint(0, T, (batch.shape[0],), device=device).long()
            
            loss = get_loss(model, batch, t, device)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "ddpm_model.pth")