import os
import sys

import numpy as np
from glob import glob
from PIL import Image


origin_img = sorted(glob('~/Datasets/railsem19/images/*.jpg'))
gt_img = sorted(glob('./outputs_background/*.jpg'))
pred_img = sorted(glob('./tmp/*.jpg'))
print(len(origin_img), len(gt_img), len(pred_img))

height, width = 1920, 1080

for idx, (ori, gt, pred) in enumerate(zip(origin_img, gt_img, pred_img), 7650):
    
    ori_img = np.array(Image.open(ori).resize((height, width)))
    gt_img = np.array(Image.open(gt).resize((height, width)))
    pred_img = np.array(Image.open(pred).resize((height, width)))

    blended_img = np.array(Image.blend(Image.fromarray(ori_img), Image.fromarray(pred_img), 0.5))

    merged_img = np.hstack((ori_img, gt_img, pred_img, blended_img))
    
    merged_img = Image.fromarray(merged_img)
    merged_img.save(f'./merged_image/rs0{idx}.png')

    