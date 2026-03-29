import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from forward_noising import forward_diffusion_sample, T
from unet import SimpleUnet
from dataloader import load_transformed_dataset


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise_pred, noise)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUnet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Multi-image defaults. Keep these isolated from single-image scripts.
    IMG_SIZE = 128
    DATASET_ROOT = "./datasets-3"
    BATCH_SIZE = 4
    EPOCHS = 1500

    dataloader = load_transformed_dataset(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        dataset_root=DATASET_ROOT,
        use_test_split=False,
        augment=False,
        repeat=1,
    )
    model.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Multi Epoch {epoch}")
        for batch, _ in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, T, (batch.shape[0],), device=device).long()
            loss = get_loss(model, batch, t, device)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "ddpm_model_multi.pth")
    print("Saved: ddpm_model_multi.pth")
