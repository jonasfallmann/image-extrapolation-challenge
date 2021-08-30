from torch.utils.data import Dataset
import numpy as np


class NormalizationProvider:
    def __init__(self, mean: float = 0.0, std: float = 0.0):
        self.mean = mean
        self.std = std
        self.no_measurements = 0

    def from_dataset(self, dataset: Dataset):
        for i in range(len(dataset)):
            img, idx = dataset.__getitem__(i)
            array = np.asarray(img, dtype=np.float32) / 255
            self.add_measurement(np.mean(array), np.std(array))
        print(self.mean, self.std)

    def add_measurement(self, m_mean: float, m_std: float):
        self.mean = (self.mean * self.no_measurements + m_mean) / (
                self.no_measurements + 1)
        self.std = (self.std * self.no_measurements + m_std) / (
                self.no_measurements + 1)
        self.no_measurements += 1

    def get_values(self):
        return self.mean, self.std
