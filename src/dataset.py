import torch
import torchvision
import torchvision.datasets as datasets

from torch.utils.data import Dataset
import numpy as np

from logger import get_logger

logger = get_logger(__name__)

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Initialize the dataset.
        :param data: List or array of input samples
        :param labels: List or array of target labels
        :param transform: Optional callable to apply transformations
        """
        self.data = data
        self.transform = transform
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        :param idx: Integer index of the sample
        :return: Tuple of (input_data, label)
        """
        image = self.data[idx][0]
        image_array = np.array(image)
        sample = torch.from_numpy(image_array)
        label = self.data[idx][1]

        # scale to [-1, 1]
        sample = sample / 255.0
        sample = sample * 2.0 - 1.0
        sample = sample.unsqueeze(0)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label   

if __name__ == "__main__":
    mnist_test = datasets.MNIST(root='../data', train=False, download=True, transform=None)
    mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=None)
    logger.info(f"Test dataset: {len(mnist_test)}")
    logger.info(f"Train dataset: {len(mnist_train)}")
    
    test_dataset = CustomDataset(mnist_test)
    train_dataset = CustomDataset(mnist_train)

    train_sample = next(iter(train_dataset))
    logger.info(f"Train sample shape: {train_sample[0].shape}")
    logger.info(f"Train sample label: {train_sample[1]}")
    logger.info(f"Min value: {train_sample[0].min()}")
    logger.info(f"Max value: {train_sample[0].max()}")