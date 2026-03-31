import os
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
import kagglehub


class BrainDataSet:
    def __init__(
        self,
        kaggle_token: str | None = None,
        img_size: tuple[int, int] = (256, 256),
        batch_size: int = 8,
        train_split: float = 0.8,
        seed: int = 42,
    ):
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_split = train_split
        self.seed = seed
        self.autotune = tf.data.AUTOTUNE

        if kaggle_token is not None:
            os.environ["KAGGLE_API_TOKEN"] = kaggle_token

        self.root_dir = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
        self.data_dir = Path(self.root_dir) / "kaggle_3m"

        self.image_paths, self.mask_paths = self._collect_paths()
        self.dataset_size = len(self.image_paths)

        if self.dataset_size == 0:
            raise ValueError(f"No data found in {self.data_dir}")

        self.train_size = int(self.train_split * self.dataset_size)
        self.val_size = self.dataset_size - self.train_size

        self.train, self.val = self._build_datasets()

    def _collect_paths(self):
        image_paths = []
        mask_paths = []

        for p in sorted(self.data_dir.rglob("*.tif")):
            if p.name.endswith("_mask.tif"):
                continue

            mask_p = p.with_name(p.stem + "_mask.tif")
            if mask_p.exists():
                image_paths.append(str(p))
                mask_paths.append(str(mask_p))

        return image_paths, mask_paths

    @staticmethod
    def _load_pair_py(image_path, mask_path):
        image_path = image_path.decode("utf-8")
        mask_path = mask_path.decode("utf-8")

        image = np.array(Image.open(image_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)

        if image.ndim == 2:
            image = image[..., None]
        if mask.ndim == 2:
            mask = mask[..., None]

        image /= 255.0
        mask /= 255.0

        return image, mask

    def _load_pair_tf(self, image_path, mask_path):
        image, mask = tf.numpy_function(
            self._load_pair_py,
            [image_path, mask_path],
            [tf.float32, tf.float32],
        )

        image.set_shape([None, None, None])
        mask.set_shape([None, None, None])

        image = tf.image.resize(image, self.img_size, method="bilinear")
        mask = tf.image.resize(mask, self.img_size, method="nearest")
        mask = tf.cast(mask > 0.5, tf.float32)

        return image, mask

    def _build_datasets(self):
        ds = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        ds = ds.shuffle(self.dataset_size, seed=self.seed, reshuffle_each_iteration=False)

        train_ds = ds.take(self.train_size).map(self._load_pair_tf, num_parallel_calls=self.autotune)
        val_ds = ds.skip(self.train_size).map(self._load_pair_tf, num_parallel_calls=self.autotune)

        train_ds = train_ds.batch(self.batch_size).prefetch(self.autotune)
        val_ds = val_ds.batch(self.batch_size).prefetch(self.autotune)

        return train_ds, val_ds

def check_for_tumors(dataset):
    tumor_count = 0
    total_samples = 0
    
    for images, masks in dataset:
        for mask in masks:
            total_samples += 1
            if tf.reduce_sum(mask) > 0:
                tumor_count += 1
    return tumor_count, total_samples
    
def get_full_dataset_tumor_counts(brain_dataset: BrainDataSet):
    """
    Calculates and prints the number of samples with and without tumors
    for the entire dataset within a BrainDataSet instance, without splitting.

    Args:
        brain_dataset_instance: An instance of the BrainDataSet class."""

    full_ds = tf.data.Dataset.from_tensor_slices((brain_dataset.image_paths, brain_dataset.mask_paths))
    full_ds = full_ds.map(brain_dataset._load_pair_tf, num_parallel_calls=brain_dataset.autotune)
    full_ds = full_ds.batch(brain_dataset.batch_size).prefetch(brain_dataset.autotune)

    full_tumor_count, full_total_samples = check_for_tumors(full_ds)
    print(f"Total samples with tumors: {full_tumor_count}")
    print(f"Total samples without tumors: {full_total_samples - full_tumor_count}")
    
def main():
    ds = BrainDataSet()
    get_full_dataset_tumor_counts(ds)

if __name__ == '__main__':
    main()