import glob

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch


class ImageExtrapolation(Dataset):

    def __init__(self, dataset_path: str, augmentation_amount: int):
        self.file_paths = glob.glob(dataset_path + "/**/*.jpg", recursive=True)
        self.file_paths += glob.glob(dataset_path + "/**/*.jpeg", recursive=True)
        self.dataset_path = dataset_path
        self.transformer = transforms.Compose([
            transforms.Resize(size=90),
            transforms.CenterCrop(size=(90, 90))
        ])
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index) -> T_co:
        path = self.file_paths[index]
        img = Image.open(os.path.join(self.dataset_path, path))

        img = self.transformer(img)
        array = np.asarray(img, dtype=np.float32)
        random_border_top = np.random.randint(4) + 4
        random_border_right = np.random.randint(4) + 4
        random_border_bottom = np.random.randint(4) + 3
        random_border_left = np.random.randint(4) + 3
        inputs, known, target = self.from_border(array, (random_border_left, random_border_right),
                                                 (random_border_top, random_border_bottom))
        return inputs, known, target, index

    def from_border(self, image_array, border_x, border_y):
        if not isinstance(image_array, np.ndarray):
            raise NotImplementedError
        if not isinstance(border_x, tuple) or not isinstance(border_y, tuple):
            raise NotImplementedError
        if image_array.ndim != 2:
            raise NotImplementedError
        if not len(border_x) == 2 or not len(border_y) == 2:
            raise ValueError
        if not isinstance(border_x[0], int) or not isinstance(border_y[0], int) or not isinstance(border_x[1],
                                                                                                  int) or not isinstance(
            border_y[1], int):
            raise ValueError
        if border_x[0] < 1 or border_x[1] < 1 or border_y[0] < 1 or border_y[1] < 1:
            raise ValueError

        width, height = image_array.shape
        remaining_width = width - sum(list(border_x))
        remaining_height = height - sum(list(border_y))
        if remaining_height < 16 or remaining_width < 16:
            raise ValueError

        input_array = np.zeros_like(image_array)
        input_array[border_x[0]:-border_x[1], border_y[0]:-border_y[1]] = image_array[border_x[0]:-border_x[1],
                                                                          border_y[0]:-border_y[1]]

        known_array = np.zeros_like(image_array)
        known_array[border_x[0]:-border_x[1], border_y[0]:-border_y[1]] = 1.0

        masked = np.copy(image_array)
        masked_signed = masked.astype('float32')
        masked_signed[border_x[0]:-border_x[1], border_y[0]:-border_y[1]] = -1
        masked_flat = masked_signed.flatten()
        target_array = image_array.flatten()[masked_flat >= 0]

        return input_array, known_array, target_array


def collate_fn(batch):
    features = torch.zeros(len(batch), 2, 90, 90).to(dtype=torch.float32)
    labels = torch.zeros(len(batch), 2475).to(dtype=torch.float32)

    for index, elem in enumerate(batch):
        features[index][0] = torch.from_numpy(batch[index][0])
        features[index][1] = torch.from_numpy(batch[index][1])
        labels[index][:len(elem[2])] = torch.from_numpy(elem[2])[:]

    return features, labels
