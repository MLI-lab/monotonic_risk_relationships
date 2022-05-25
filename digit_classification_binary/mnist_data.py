import numpy as np

from torch.utils.data import Dataset

from utilities import get_datetime


class MNISTDataset(Dataset):
    """
    """
    def __init__(self, config, label_map=None, transform=None):
        """
        """
        self.label_map = label_map
        self.transform = transform
        self.image_directory = config["image_directory"]
        self.label_directory = config["label_directory"]
        self.split = config["split"]
        # self.normalization = config["normalization"]

        self.load_images_and_labels()

    def load_images_and_labels(self):
        """
        """
        image_size = 28
        label_size = 1
        image_skip_bytes = 16 # the fist 16 bytes are for meta information
        label_skip_bytes = 8  # the fist 8 bytes are for meta information

        image_skip_bytes = image_skip_bytes + image_size**2 * self.split["begin"] 
        label_skip_bytes = label_skip_bytes + label_size * self.split["begin"]

        with open(self.image_directory, 'rb') as f:
            f.read(image_skip_bytes)
            buf = f.read(image_size**2 * self.split["num_samples"])
            self.images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            self.images = self.images.reshape(self.split["num_samples"], image_size, image_size, 1) # (N, H, W, 1)
            # self.images = self.images.transpose((0, 3, 1, 2))
            # self.images = (self.images - np.asarray(self.normalization["mean"])[None, :, None, None])
            # self.images =  self.images / np.asarray(self.normalization["std"])[None, :, None, None]
        with open(self.label_directory, 'rb') as f:
            f.read(label_skip_bytes)
            buf = f.read(label_size * self.split["num_samples"])
            self.labels = np.frombuffer(buf, dtype=np.uint8).astype(int)
        if self.label_map is not None:
            self.labels = np.fromiter(map(lambda x: self.label_map[x], self.labels), dtype=np.int)
            # self.labels = np.fromiter(map(lambda x: self.label_map[x], self.labels), dtype=np.float32)
        print("{} {} images and {} labels in {} loaded.".format(get_datetime(), len(self.images), len(self.labels), np.unique(self.labels).tolist()))

    def __len__(self):
        """
        """
        return len(self.labels)

    def __getitem__(self, i):
        """
        """
        sample = dict()
        image = self.images[i]
        label = self.labels[i]
        if self.transform is not None:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label

        return sample
