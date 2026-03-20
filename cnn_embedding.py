import os
import keras
import numpy as np # Use numpy for the data loader
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from keras.applications.resnet_v2 import ResNet50V2
from tqdm import tqdm


# Constants
OUTPUT_FILENAME = r"maps/cnn_embedding_gray_v4.npz"
DATA_PATH = 'data/Galaxy10_ProcessedandCroppedV4.h5'
BATCH_SIZE = 64 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.length = f['images'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:

            img = f['images'][idx].astype(np.float32)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            img = (img / 127.5) - 1.0 

            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            img_tensor = img_tensor.expand(3, -1, -1)
            
            return img_tensor

def run_extraction():

    model = ResNet50V2(
            weights='imagenet', 
            include_top=False, 
            pooling='avg', 
            input_shape=(128, 128, 3)
    )

    dataset = HDF5Dataset(DATA_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,   
        pin_memory=True  
    )

    all_features = []

    for batch in tqdm(dataloader, desc="Extracting Features"):

        batch_np = batch.permute(0, 2, 3, 1).cpu().numpy() 
        preds = model(batch_np, training=False) 
        all_features.append(preds)

    return np.vstack(all_features)

if __name__ == "__main__":

    cnn_embeddings = run_extraction()

    np.savez_compressed(OUTPUT_FILENAME, cnn_embeddings=cnn_embeddings)