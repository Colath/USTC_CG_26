from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os


def build_transforms(img_size=256, augment=False):
    data_transforms = [transforms.Resize((img_size, img_size))]
    if augment:
        data_transforms.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    data_transforms.extend(
        [
            transforms.ToTensor(),  # Scales data into [0,1]
            transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ]
    )
    return transforms.Compose(data_transforms)


def load_dataset_with_metadata(img_size=256, dataset_root="./datasets-1", use_test_split=True, augment=False):
    data_transform = build_transforms(img_size=img_size, augment=augment)
    train = torchvision.datasets.ImageFolder(root=os.path.join(dataset_root, "train"), transform=data_transform)
    datasets = [train]

    test_root = os.path.join(dataset_root, "test")
    if use_test_split and os.path.exists(test_root):
        datasets.append(torchvision.datasets.ImageFolder(root=test_root, transform=data_transform))

    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    classes = list(train.classes)
    class_to_idx = dict(train.class_to_idx)
    return dataset, classes, class_to_idx


def load_transformed_dataset(
    img_size=256,
    batch_size=128,
    dataset_root="./datasets-1",
    use_test_split=True,
    augment=False,
    repeat=1,
) -> DataLoader:
    dataset, _, _ = load_dataset_with_metadata(
        img_size=img_size,
        dataset_root=dataset_root,
        use_test_split=use_test_split,
        augment=augment,
    )

    if repeat > 1:
        dataset = torch.utils.data.ConcatDataset([dataset] * repeat)

    # Small datasets should not drop the only batch.
    drop_last = len(dataset) >= batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)


def show_tensor_image(image):
    # Reverse the data transformations
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))