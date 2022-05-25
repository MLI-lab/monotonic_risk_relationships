import numpy as np

from torch.utils.data import Dataset

from utilities import get_datetime


class ARDISDataset(Dataset):
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
        self.images = np.loadtxt(self.image_directory, dtype="float")
        self.labels = np.loadtxt(self.label_directory, dtype="float")

        self.images = self.images.reshape(self.images.shape[0], 28, 28).astype("float32")
        self.labels = np.argmax(self.labels, axis=1)

        if self.label_map is not None:
            self.labels = np.fromiter(map(lambda x: self.label_map[x], self.labels), dtype=np.int)
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
