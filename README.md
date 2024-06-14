# Railsem19 Semantic Segmentation with DeepLabV3Plus

## Project Goal

- The goal of this project is to perform semantic segmentation on the Railsem19 dataset to classify rail and track, as well as to detect other railway-related structures.
- This will help evaluate and improve object detection performance in various railway environments.

## Model Reference

- This project uses the DeepLabV3Plus model.
- For more details on the model, visit the following repository: [DeepLabV3Plus for Beginners](https://github.com/J911/DeepLabV3Plus-for-Beginners).

## Dataset

- The dataset used in this project is the Railsem19 dataset.
- For more information on the dataset, visit: [Railsem19 Dataset](https://www.wilddash.cc/railsem19).

## Experiments

Three experiments were conducted in this project:
1. Training only 8 classes excluding the background class.
2. Training only 8 classes including the background class.
3. Training with all labels including the background class.

## Sample Images

Below are some sample images from the dataset and the segmentation results:

- 8 classes excluding the background class
![Sample Image 1](outputs/good_without_background/rs07869.png)
- 8 classes including the background class.
![Sample Image 2](outputs/good_background/rs07854.png)
- All labels including the background class.
![Sample Image 3](outputs/good_full_labels/rs07855.png)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/railsem19-semantic-segmentation.git
   cd railsem19-semantic-segmentation


## Train
git clone & change DIR
```bash
$ git clone https://github.com/J911/DeepLabV3Plus-for-Beginners
$ cd DeepLabV3Plus-for-Beginners
```
run 🙌🙌
```bash
$ python train.py --data /data/CITYSCAPES --batch-size 16 --epoch 200 --logdir ./logs/exp1/ --save ./saved_model/exp1/
```

## Evaluate
```bash
$ python evaluate.py --data /data/CITYSCAPES --weight ./saved_model/exp1/epoch200.pth --num-classes 19
```

## Dataset

This Repository uses Cityscapes Dataset.

```
CITYSCAPES
|-- leftImg8bit
|   |-- test 
|   |-- train
|   `-- val
`-- gtFine
    |-- test 
    |-- train
    `-- val
```

## Result

- Encoder: ResNet101-OS16
- LR: 1e-2 ~ 1e-4 (CosineAnnealingLR)
- Weight Decay: 5e-4
- Epoch: 200
- Batch Size: 16 (8 images per GPU)
- GPU: 2GPU (Tesla V100 * 2)

- mIoU: 0.7521

![result](./assets/result.png)

| Class | IoU | Class | IoU | Class | IoU | Class | IoU |
|:-----:|:---:|:-----:|:---:|:-----:|:---:|:-----:|:---:|
| **road** | 0.9823 | **pole** | 0.6408 | **sky** | 0.9455 | **bus** | 0.8117 |
| **sidewalk** | 0.8528 | **traffic light** | 0.6935 | **person** | 0.8175 | **train** | 0.5439 |
| **building** | 0.9215 | **traffic sign** | 0.7805 | **rider** | 0.6328 | **motorcycle** | 0.6905 |
| **wall** | 0.4955 | **vegetation** | 0.9245 | **car** | 0.9445 | **bicycle** | 0.7738 |
| **fence** | 0.5871 | **terrain** | 0.6148 | **truck** | 0.6354 | - | - |


## Thanks to
@speedinghzl - Gain a lot of Insight 🙇🏻‍♂️
