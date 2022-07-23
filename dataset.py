import os 
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import re

class KITTIDataset(Dataset):
    def __init__(self, root=None, mode="train", cropsize = None):
        self.root = root
        self.mode = mode
        self.trans = transforms.ToTensor()
        self.crop = transforms.CenterCrop((368,560)) if not mode=="test" else transforms.CenterCrop((352,1216))
        if cropsize != None:
            self.crop = transforms.CenterCrop((368, cropsize))
        self.data = []
        if mode=="train":
            raw_path = [p for p in glob.glob(root + "data_depth_velodyne/train/**", recursive=True) if re.search("2011_09_26_drive_\d{4}_sync/proj_depth/velodyne_raw/image_02/00000000\d{2}.png", p)]
            names = [p.split('/') for p in raw_path]
            gt_path = ['/'.join((*p[:-7],"data_depth_annotated",*p[-6:-3],"groundtruth",*p[-2:])) for p in names]
            data = [[raw_path[i], gt_path[i]] for i in range(len(raw_path))]
            self.data.extend(data)
        if mode=="val":
            raw_path = [p for p in glob.glob(root + "data_depth_velodyne/val/**", recursive=True) if re.search("2011_09_26_drive_\d{4}_sync/proj_depth/velodyne_raw/image_02/00000000\d{2}.png", p)]
            names = [p.split('/') for p in raw_path]
            gt_path = ['/'.join((*p[:-7],"data_depth_annotated",*p[-6:-3],"groundtruth",*p[-2:])) for p in names]
            data = [[raw_path[i], gt_path[i]] for i in range(len(raw_path))]
            self.data.extend(data)
        if mode=="test":
            #img_path = [p for p in glob.glob(root + "depth_selection/test_depth_completion_anonymous/**", recursive=True) if re.search("image/00000000\d{2}.png", p)]
            #names = [p.split('/') for p in img_path]
            #raw_path = ['/'.join((*p[:3],"velodyne_raw",p[4])) for p in names]
            #data = [[img_path[i], raw_path[i]] for i in range(len(img_path))]
            #self.data.extend(data)
            img_path = [p for p in glob.glob(root + "depth_selection/val_selection_cropped/**", recursive=True) if re.search("image/2011_\d{2}_\d{2}_drive_\d{4}_sync_image_\d{10}_image_\d{2}.png", p)]
            names = [[p.split('/'), p.split('/')[-1].split('_')]  for p in img_path]
            raw_path = ['/'.join((*p[0][:-2],"velodyne_raw",'_'.join((*p[1][:6],"velodyne_raw",*p[1][7:])))) for p in names]
            gt_path = ['/'.join((*p[0][:-2],"groundtruth_depth",'_'.join((*p[1][:6],"groundtruth_depth",*p[1][7:])))) for p in names]
            data = [[img_path[i], raw_path[i], gt_path[i]] for i in range(len(img_path))]
            self.data.extend(data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.mode=="train" or self.mode=="val":
            img1 = np.array(Image.open(self.data[index][0]), dtype=int)
            img1 = img1.astype(np.float32) / 256.
            img1 = np.expand_dims(img1, 0)
            img1 = torch.from_numpy(img1.astype(np.float32)).clone()
            img1 = self.crop(img1)
            img2 = np.array(Image.open(self.data[index][1]), dtype=int)
            img2 = img2.astype(np.float) / 256.
            img2 = np.expand_dims(img2, 0)
            img2 = torch.from_numpy(img2.astype(np.float32)).clone()
            img2 = self.crop(img2)
            return [img1, img2]
        if self.mode=="test":
            img1 = Image.open(self.data[index][0]).convert('RGB')
            img1 = np.array(img1, dtype=int)
            img2 = np.array(Image.open(self.data[index][1]), dtype=int)
            img2 = img2.astype(np.float) / 256.
            img2 = np.expand_dims(img2, 0)
            img2 = torch.from_numpy(img2.astype(np.float32)).clone()
            img2 = self.crop(img2)
            img3 = np.array(Image.open(self.data[index][2]), dtype=int)
            img3 = img3.astype(np.float) / 256.
            img3 = np.expand_dims(img3, 0)
            img3 = torch.from_numpy(img3.astype(np.float32)).clone()
            img3 = self.crop(img3)
            return [img1, img2, img3]
    
    def __repr__(self):
        return f"Number of Data: {len(self)}\nRoot: {self.root}\nMode: {self.mode}"