import os
from PIL import Image
from torch.utils.data import Dataset
import kagglehub

class Food11Dataset(Dataset):
    def __init__(self, dataset_option = 'training', transform=None):
        os.environ["KAGGLE_API_TOKEN"] = "ENTER_YOUR_TOKEN"
        self.root_dir = kagglehub.dataset_download("vermaavi/food11")
        print("Downloaded to:", self.root_dir)
        self.image_path = os.path.join(self.root_dir, dataset_option)

        self.transform = transform

        self.files = sorted([
            f for f in os.listdir(self.image_path)
            if f.endswith(".jpg")
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        label = int(fname.split("_")[0])

        img_path = os.path.join(self.image_path, fname)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label