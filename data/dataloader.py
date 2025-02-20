import os
import numpy as np
import random
import torch
import torchvision
import cv2
from torch.utils import data
import glob

class DataSet(data.Dataset):
    def __init__(self, root, train=True, input_size=(768, 768), std=[0.485, 0.456, 0.406], mean=[0.229, 0.224, 0.225], mirror=True, ignore_label=255):
        self.root = root
        self.mean = mean
        self.mirror = mirror
        self.ignore_label = ignore_label
        self.input_size = input_size

        self.std = std
        self.mean = mean

        self.train = train

        if self.train == True:
            self.image_dir = os.path.join(self.root, 'train/images')
            self.label_dir = os.path.join(self.root, 'train/labels')
        else:
            self.image_dir = os.path.join(self.root, 'val/images')
            self.label_dir = os.path.join(self.root, 'val/labels')

        self.image_paths = glob.glob(self.image_dir + '*/*.jpg')
        self.label_paths = glob.glob(self.label_dir + '*/*.png')

        self.image_paths.sort()
        self.label_paths.sort()

        # self.id_to_trainid = {0:0, 1:1, 2:2, 3:3, 4:4,
        #                       5:5, 6:6, 7:7, 8:8, 9:9,
        #                       10:10, 11:11, 12:12, 13:13, 14:14,
        #                       15:15, 16:16, 17:17, 18:18,
        #                       255: 255}
        
        self.id_to_trainid = {0:ignore_label, 1:ignore_label, 2:ignore_label, 3:3, 4:ignore_label,
                              5:ignore_label, 6:6, 7:7, 8:ignore_label, 9:ignore_label,
                              10:ignore_label, 11:11, 12:12, 13:ignore_label, 14:ignore_label,
                              15:ignore_label, 16:16, 17:17, 18:18,
                              255: ignore_label}

        # training class: tram-track(3), traffic-light(6), traffic-sign(7), human(11), rail-track(12), on-rails(16), rail-raised(17), rail-embeded(18)

    def id2trainId(self, label):
        label_copy = label.copy()
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        label = cv2.imread(self.label_paths[index], cv2.IMREAD_GRAYSCALE)

        image = image / 255.0
        image = image - np.array(self.std)
        image = image / np.array(self.mean)

        if self.train == True:
            scale = np.random.uniform(low=0.75, high=2.0)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        

        h, w = self.input_size

        h0 = random.randint(0, label.shape[0] - h)
        w0 = random.randint(0, label.shape[1] - w)

        image = image[h0:h0+h, w0:w0+w]
        label = label[h0:h0+h, w0:w0+w]

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label = self.id2trainId(label)

        image = image.transpose((2, 0, 1))

        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy()