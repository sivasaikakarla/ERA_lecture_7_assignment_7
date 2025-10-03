# CIFAR-10 Image Classification with PyTorch

This project contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The implementation includes data augmentation using Albumentations, a custom model architecture, and a complete training and evaluation pipeline.

## 🚀 Features

-   **Dataset**: CIFAR-10
-   **Framework**: PyTorch
-   **Augmentations**: Albumentations library for robust data augmentation (HorizontalFlip, ShiftScaleRotate, CoarseDropout).
-   **Model**: Custom CNN architecture featuring Depthwise Separable Convolutions, Dilated Convolutions, and a Global Average Pooling (GAP) layer.
-   **Training**: Includes a full training and testing loop with progress bars (tqdm) and performance logging (PrettyTable).

## 📂 File Structure

```
.
├── model.py      # Contains the CIFAR10Net class definition.
├── train.py      # Main script to handle data loading, training, and evaluation.
└── README.md     
```

## 🛠️ Setup and Installation


1.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install torch torchvision torchaudio
    pip install numpy albumentations prettytable torchsummary tqdm
    ```

## 🏃 How to Run

To start the training process, simply run the `train.py` script from your terminal:

```bash
python train.py
```

The script will automatically download the CIFAR-10 dataset, build the model, and begin training for 50 epochs. The progress and results will be printed to the console.

## 📊 Model Architecture Summary

The model architecture is composed of several convolutional blocks designed to efficiently extract features from the input images.

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
              ReLU-2           [-1, 16, 32, 32]               0
       BatchNorm2d-3           [-1, 16, 32, 32]              32
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,608
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 32, 16, 16]           9,216
             ReLU-10           [-1, 32, 16, 16]               0
      BatchNorm2d-11           [-1, 32, 16, 16]              64
           Conv2d-12           [-1, 32, 16, 16]             288
           Conv2d-13           [-1, 64, 16, 16]           2,048
             ReLU-14           [-1, 64, 16, 16]               0
      BatchNorm2d-15           [-1, 64, 16, 16]             128
          Dropout-16           [-1, 64, 16, 16]               0
           Conv2d-17           [-1, 64, 16, 16]          36,864
             ReLU-18           [-1, 64, 16, 16]               0
      BatchNorm2d-19           [-1, 64, 16, 16]             128
          Dropout-20           [-1, 64, 16, 16]               0
           Conv2d-21            [-1, 128, 8, 8]          73,728
             ReLU-22            [-1, 128, 8, 8]               0
      BatchNorm2d-23            [-1, 128, 8, 8]             256
AdaptiveAvgPool2d-24            [-1, 128, 1, 1]               0
           Conv2d-25             [-1, 10, 1, 1]           1,280
================================================================
Total params: 129,136
Trainable params: 129,136
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.94
Params size (MB): 0.49
Estimated Total Size (MB): 3.44
----------------------------------------------------------------
```

## 📈 Final Results

The model was trained for 50 epochs, achieving the following final performance:

-   **Training Accuracy**: `90.05%`
-   **Test Accuracy**: `85.40%`

Below is the complete log of the training process:

| Epoch | Training Accuracy | Test Accuracy |  Diff  | Training Loss | Test Loss |
| :---: | :---------------: | :-----------: | :----: | :-----------: | :-------: |
|   1   |       49.58%      |     60.57%    | -10.99 |     1.2238    |   1.0870  |
|   2   |       64.34%      |     68.00%    | -3.66  |     0.9151    |   0.9155  |
|   3   |       70.18%      |     72.81%    | -2.63  |     0.9749    |   0.7707  |
|   4   |       73.57%      |     74.77%    | -1.20  |     0.7730    |   0.7294  |
|   5   |       76.14%      |     76.25%    | -0.11  |     0.5815    |   0.6831  |
|   6   |       77.98%      |     78.90%    | -0.92  |     0.8226    |   0.6047  |
|   7   |       79.31%      |     79.40%    | -0.09  |     0.7528    |   0.5960  |
|   8   |       80.24%      |     79.88%    |  0.36  |     0.6128    |   0.5838  |
|   9   |       81.17%      |     81.29%    | -0.12  |     0.6029    |   0.5375  |
|   10  |       81.67%      |     82.05%    | -0.38  |     0.3952    |   0.5178  |
|   11  |       82.56%      |     82.10%    |  0.46  |     0.5979    |   0.5259  |
|   12  |       82.93%      |     82.42%    |  0.51  |     0.3592    |   0.5172  |
|   13  |       83.33%      |     82.81%    |  0.52  |     0.5130    |   0.5061  |
|   14  |       83.77%      |     83.23%    |  0.54  |     0.6214    |   0.4939  |
|   15  |       84.30%      |     83.26%    |  1.04  |     0.5182    |   0.4951  |
|   16  |       84.60%      |     82.51%    |  2.09  |     0.3966    |   0.5275  |
|   17  |       84.96%      |     83.37%    |  1.59  |     0.3876    |   0.4907  |
|   18  |       85.35%      |     82.90%    |  2.45  |     0.4424    |   0.5149  |
|   19  |       85.49%      |     83.01%    |  2.48  |     0.4199    |   0.5059  |
|   20  |       85.84%      |     83.93%    |  1.91  |     0.3962    |   0.4948  |
|   21  |       86.06%      |     84.32%    |  1.74  |     0.3928    |   0.4611  |
|   22  |       86.19%      |     83.66%    |  2.53  |     0.5805    |   0.4875  |
|   23  |       86.54%      |     83.53%    |  3.01  |     0.4518    |   0.4880  |
|   24  |       86.57%      |     84.26%    |  2.31  |     0.2597    |   0.4722  |
|   25  |       87.00%      |     84.10%    |  2.90  |     0.3585    |   0.4892  |
|   26  |       87.38%      |     84.43%    |  2.95  |     0.2123    |   0.4682  |
|   27  |       87.15%      |     84.46%    |  2.69  |     0.6947    |   0.4675  |
|   28  |       87.43%      |     83.90%    |  3.53  |     0.4666    |   0.4910  |
|   29  |       87.71%      |     84.65%    |  3.06  |     0.2916    |   0.4846  |
|   30  |       87.79%      |     84.96%    |  2.83  |     0.4124    |   0.4545  |
|   31  |       88.04%      |     84.98%    |  3.06  |     0.6807    |   0.4524  |
|   32  |       88.19%      |     84.51%    |  3.68  |     0.3112    |   0.4653  |
|   33  |       88.15%      |     84.54%    |  3.61  |     0.4190    |   0.4791  |
|   34  |       88.12%      |     84.34%    |  3.78  |     0.2844    |   0.4733  |
|   35  |       88.57%      |     85.19%    |  3.38  |     0.5898    |   0.4586  |
|   36  |       88.56%      |     84.54%    |  4.02  |     0.3732    |   0.4617  |
|   37  |       88.73%      |     84.65%    |  4.08  |     0.4005    |   0.4684  |
|   38  |       88.92%      |     84.60%    |  4.32  |     0.4068    |   0.4705  |
|   39  |       88.95%      |     85.25%    |  3.70  |     0.3579    |   0.4563  |
|   40  |       88.97%      |     85.30%    |  3.67  |     0.3400    |   0.4537  |
|   41  |       89.03%      |     85.61%    |  3.42  |     0.2314    |   0.4453  |
|   42  |       89.54%      |     85.25%    |  4.29  |     0.4962    |   0.4596  |
|   43  |       89.08%      |     84.24%    |  4.84  |     0.3248    |   0.4859  |
|   44  |       89.52%      |     84.82%    |  4.70  |     0.3755    |   0.4852  |
|   45  |       89.61%      |     85.40%    |  4.21  |     0.3704    |   0.4560  |
|   46  |       89.44%      |     84.87%    |  4.57  |     0.2555    |   0.4606  |
|   47  |       89.75%      |     84.69%    |  5.06  |     0.2557    |   0.4732  |
|   48  |       89.71%      |     84.78%    |  4.93  |     0.3020    |   0.4761  |
|   49  |       89.77%      |     85.40%    |  4.37  |     0.3766    |   0.4630  |
|   50  |       90.05%      |     85.38%    |  4.67  |     0.4159    |   0.4559  |
