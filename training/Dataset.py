import os
from torch.utils.data import Dataset
import zstandard as zstd
import pickle
import numpy as np

class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        if split not in ['train', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of: 'train', 'test'.")
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        """
        Finds the class names and creates a mapping from class name to index.
        """
        class_names = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        return class_names, class_to_idx

    def _make_dataset(self):
        """
        Collects all sample file paths and their corresponding labels.
        """
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name, self.split)
            if not os.path.isdir(class_dir):
                continue
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    if fname.endswith('.zst'):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[class_name])
                        samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        voxel_grid = self.deserialize_and_decompress(path)

        if self.transform is not None:
            voxel_grid = self.transform(voxel_grid)

        return voxel_grid, target

    @staticmethod
    def deserialize_and_decompress(zst_input_path):
        """
        Decompresses and deserializes the voxel grid from the given .zst file.
        """
        dctx = zstd.ZstdDecompressor()
        with open(zst_input_path, 'rb') as f:
            decompressed_data = dctx.decompress(f.read())
            filled_voxel_grid = pickle.loads(decompressed_data)

        # Convert the data to integers if necessary
        filled_voxel_grid = filled_voxel_grid.astype("float32")
        filled_voxel_grid = np.expand_dims(filled_voxel_grid, axis=0) # Add channel dimension

        return filled_voxel_grid
    
    def get_class_mapping(self):
        return self.class_to_idx