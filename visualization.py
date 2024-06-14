import torch
import torch.nn as nn

from data.dataloader_eval import DataSet
from models.deeplabv3plus import DeepLabV3Plus
import torch.nn.functional as F
import argparse

import numpy as np
import os
import sys
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
parser.add_argument("--data", type=str, default="/dataset", help="")
parser.add_argument("--weight", type=str, default="./saved_model/full_label_best.pth", help="")
parser.add_argument("--num-classes", type=int, default=20, help="")

args = parser.parse_args()

print(args)

batch_size = 1

test_dataset = DataSet(args.data, train=False, input_size=(1920, 1080), mirror=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = DeepLabV3Plus(num_classes=args.num_classes)
net = net.to(device)

checkpoint = torch.load(args.weight)
net.load_state_dict(checkpoint['net'])

# Define colormap for each label
colormap = {
    0: [128, 64, 128],
    1: [224, 35, 232],
    2: [70, 70, 70],
    3: [192, 0, 128],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180],
    11: [220, 20, 60],
    12: [230, 150, 140],
    13: [0, 0, 142],
    14: [0, 0, 70],
    15: [90, 40, 40],
    16: [0, 80, 100],
    17: [0, 254, 254],
    18: [0, 68, 63],
    # 19: [0, 0, 0],
}

# not_train = list(set(idx for idx in range(20)).difference(set(colormap.keys())))
not_train = [255]
print(not_train)

def get_confusion_matrix(gt_label, pred_label, class_num):
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
        for i_pred_label in range(class_num):
            cur_index = i_label * class_num + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def test():
    net.eval()
    
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    
    for images, labels, path, gt in tqdm(test_loader):
        
        _, h, w = labels.size()
        images = images.to(device)
        labels = labels.to(device)
        out = net(images)
        out = F.interpolate(out, size=(h, w), mode='bilinear')

        out = torch.argmax(out, dim=1).squeeze()
        labels = labels.squeeze()
        
        out = out.cpu().numpy()
        labels = labels.cpu().numpy()    

        gt = gt.squeeze()

        # case2
        # vis_img2 = [[[] for _ in range(out.shape[1])] for _ in range(out.shape[0])]
         
        # for i in tqdm(range(out.shape[0])):
        #     for j in range(out.shape[1]):
        #         # print(i, j, out[i][j], labels[i][j], '||', colormap[out[i][j]])
        #         if labels[i][j] in colormap.keys():
        #             vis_img2[i][j] = colormap[out[i][j]]
        #         else:
        #             vis_img2[i][j] = [0, 0, 0]
                    
        # vis_img2 = np.array(vis_img2)
        # pil_image = Image.fromarray(vis_img2.astype(np.uint8))
        
         
        # case3
        vis_img_pred = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
        
        for label, color in colormap.items():
            vis_img_pred[out == label] = color
        
        for idx in not_train:
            vis_img_pred[gt == idx] = (0, 0, 0)
        
        pil_image = Image.fromarray(vis_img_pred)
        pil_image.save(f"./outputs_full/{os.path.basename(path[0])}")
        

        vis_img_gt = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)    
        for label, color in colormap.items():
            vis_img_gt[gt == label] = color    
        pil_image = Image.fromarray(vis_img_gt)
        pil_image.save(f"./gt_full/{os.path.basename(path[0])}")
        

if __name__=='__main__':
    test()
