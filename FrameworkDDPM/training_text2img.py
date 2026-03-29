import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from cond_unet import ConditionalUnet
from dataloader import load_dataset_with_metadata
from forward_noising import forward_diffusion_sample, T


def get_loss(model, x_0, class_ids, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t, class_ids)
    return F.mse_loss(noise_pred, noise)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    IMG_SIZE = 128
    DATASET_ROOT = "./datasets-2"
    BATCH_SIZE = 8
    EPOCHS = 2000

    dataset, classes, class_to_idx = load_dataset_with_metadata(
        img_size=IMG_SIZE,
        dataset_root=DATASET_ROOT,
        use_test_split=False,
        augment=True,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = ConditionalUnet(num_classes=len(classes)).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Text2Img Epoch {epoch}")
        for batch, class_ids in pbar:
            batch = batch.to(device)
            class_ids = class_ids.to(device).long()
            optimizer.zero_grad()
            t = torch.randint(0, T, (batch.shape[0],), device=device).long()
            loss = get_loss(model, batch, class_ids, t, device)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "ddpm_model_text2img.pth")
    with open("text2img_classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, ensure_ascii=False, indent=2)

    print("Saved: ddpm_model_text2img.pth")
    print("Saved: text2img_classes.json")
