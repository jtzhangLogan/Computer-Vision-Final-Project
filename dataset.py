"""
Our implementation
"""
import os
import numpy as np
import torch
from skimage.io import imread
from skimage.transform import rescale
from torch.utils.data import Dataset, DataLoader
import natsort
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class SurgicalVisDomDataset(Dataset):
    def __init__(self, images_dir, subset, transform=None):

        assert subset in ["train", "test"]

        self.subset = subset
        self.surgery_im = []
        self.vr_im = []
        self.transform = transform

        print("Creating {} Datasets...".format(subset))

        if subset == "train":
            vr_path = images_dir + "/train_1/VR/Dissection/captures"
            surgery_path = images_dir + "/train_1/Porcine/Dissection/captures"
        else:
            vr_path = images_dir + "/test/VR/Dissection/captures"
            surgery_path = images_dir + "/test/Porcine/Dissection/captures"

        for filename in natsort.natsorted(os.listdir(vr_path), reverse=False):
            self.vr_im.append(
                np.array(rescale(imread(os.path.join(vr_path, filename)), (0.2, 0.2), anti_aliasing=True)))

        for filename in natsort.natsorted(os.listdir(surgery_path), reverse=False):
            self.surgery_im.append(
                np.array(rescale(imread(os.path.join(surgery_path, filename)), (1/3.75, 1/3.75), anti_aliasing=True)))

        print("Done creating {} dataset".format(subset))

    def __len__(self):
        return max(len(self.surgery_im), len(self.vr_im))

    def __getitem__(self, idx):
        if self.subset == "train":
            vr_list = [self.vr_im[idx], self.vr_im[idx + 1], self.vr_im[idx + 2]]
            surgery_list = [self.surgery_im[idx], self.surgery_im[idx + 1], self.surgery_im[idx + 2]]
        else:
            vr_list = [self.vr_im[idx]]
            surgery_list = [self.surgery_im[idx]]

        # fix dimensions (C, H, W)
        for i in range(len(vr_list)):
            vr_list[i] = torch.from_numpy(vr_list[i].transpose(2, 0, 1).astype(np.float32))
        for i in range(len(surgery_list)):
            surgery_list[i] = torch.from_numpy(surgery_list[i].transpose(2, 0, 1).astype(np.float32))

        if self.transform is not None:
            for i in range(len(vr_list)):
                vr_list[i] = self.transform(vr_list[i])
            for i in range(len(surgery_list)):
                surgery_list[i] = self.transform(surgery_list[i])

        # return tensors
        return vr_list, surgery_list


if __name__ == '__main__':

    transform = None

    train = SurgicalVisDomDataset('./input_data', "train", transform=transform)

    loader_train = DataLoader(
        train,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )

    loaders = {"train": loader_train}

    for k, data in enumerate(loaders["train"]):
        vr, surgery = data

        for i in range(len(vr)):
            plt.figure(i)
            plt.imshow(vr[i].cpu().detach().numpy()[0].transpose(1, 2, 0))
            plt.show()

        for i in range(len(surgery)):
            plt.figure(i + 3)
            plt.imshow(surgery[i].cpu().detach().numpy()[0].transpose(1, 2, 0))
            plt.show()

        break
