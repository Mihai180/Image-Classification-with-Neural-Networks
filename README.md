# Image Classification with Neural Networks

This project is a Computer Vision assignment focused on image classification using neural networks in PyTorch.

The goal of the project is to build, train, evaluate and compare multiple neural network approaches for two image classification problems:

* skin lesion classification
* facial emotion recognition

The project covers the complete training pipeline, including custom datasets, dataloaders, data augmentation, MLP and CNN architectures, transfer learning, class imbalance handling and ablation studies.

## Project Overview

This assignment explores the use of neural networks for image classification tasks.
The implementation follows a full deep learning workflow:

1. Dataset loading and preprocessing
2. Image augmentation
3. MLP training and evaluation
4. CNN training and evaluation
5. Transfer learning using pretrained models
6. Class imbalance handling
7. Experiment tracking
8. Model comparison through ablation studies

The project was developed using PyTorch and focuses on understanding how different architectural choices and training strategies affect model performance.

## Datasets

The project uses two image classification datasets.

### Skin Lesion Classification

The first dataset contains dermatoscopic images of pigmented skin lesions.
The goal is to classify each image into one of several diagnostic categories.

Classes:

* `akiec`
* `bcc`
* `bkl`
* `df`
* `mel`
* `nv`
* `vasc`

The dataset contains RGB images and is imbalanced, meaning that some lesion types appear much more frequently than others.

### Facial Emotion Recognition

The second dataset contains facial images labeled with basic emotions.

Classes:

* Surprise
* Fear
* Disgust
* Happiness
* Sadness
* Anger
* Neutral

This dataset is also imbalanced and contains images captured in real-world conditions, with variations in lighting, pose, occlusion and facial expression intensity.

## Main Features

* Custom PyTorch Dataset and DataLoader pipeline
* Image resizing and normalization
* Data augmentation for both datasets
* MLP architecture for image classification
* CNN architecture trained from scratch
* Batch Normalization experiments
* Dropout experiments
* Transfer learning with pretrained models
* Learning rate warmup experiments
* Class imbalance handling
* Accuracy, F1-macro and F1-micro evaluation
* Confusion matrix generation
* Train/test loss and accuracy curves
* TensorBoard experiment logging
* Ablation studies for model comparison

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorBoard


## Neural Network Models

### MLP

A Multi-Layer Perceptron was implemented as a baseline neural architecture.

The MLP receives flattened image tensors as input and uses fully connected layers with nonlinear activations.

Experiments include:

* MLP without dropout
* MLP with dropout
* different hidden layer sizes
* different learning rates

The goal of this model is to establish a simple baseline and compare it with convolutional architectures.

### CNN

A Convolutional Neural Network was implemented and trained from scratch.

The CNN uses convolutional layers to extract spatial features from images, followed by pooling and fully connected classification layers.

Experiments include:

* CNN without Batch Normalization
* CNN with Batch Normalization
* CNN with different augmentation strategies
* CNN with class imbalance handling

The CNN architecture follows the assignment restrictions:

* maximum 4 convolutional layers
* maximum 2 fully connected layers

### Transfer Learning

Transfer learning was applied using pretrained computer vision models.

Possible pretrained architectures:

* ResNet-18
* MobileNetV2

The final classification layer was replaced to match the number of classes in each dataset.

Experiments include:

* frozen backbone with trainable classification head
* fine-tuning with different learning rates
* fine-tuning with learning rate warmup
* comparison with models trained from scratch

## Data Augmentation

Data augmentation was used to improve generalization and reduce overfitting.

### Geometric Augmentations

Examples:

* random horizontal flip
* random rotation
* random crop
* resize
* random scaling

### Color Augmentations

Examples:

* brightness adjustment
* contrast adjustment
* saturation adjustment
* color jitter
* blur

For skin lesion images, color augmentations were kept moderate because color information can be relevant for diagnosis.

For facial emotion images, geometric transformations were chosen carefully so that they do not change the semantic meaning of the facial expression.

## Handling Class Imbalance

Both datasets contain imbalanced class distributions.

The following techniques were considered:

* weighted cross entropy loss
* weighted random sampling
* monitoring F1-macro instead of only accuracy
* confusion matrix analysis
* per-class performance evaluation

This is important because accuracy alone can be misleading when some classes are much more frequent than others.

## Training Pipeline

The training pipeline includes:

1. Loading the dataset
2. Applying transformations and augmentations
3. Creating train and validation dataloaders
4. Initializing the model
5. Choosing the loss function
6. Choosing the optimizer
7. Training for multiple epochs
8. Evaluating on the validation set
9. Saving the best model checkpoint
10. Logging metrics and plots

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* F1-macro
* F1-micro
* Confusion matrix
* Train loss
* Validation loss
* Train accuracy
* Validation accuracy

F1-macro is especially important because both datasets are imbalanced.

## Ablation Studies

The project compares different configurations through ablation studies.

Main ablations:

### MLP

| Experiment    | Dropout | Accuracy | F1-macro | F1-micro |
| ------------- | ------: | -------: | -------: | -------: |
| MLP baseline  |      No |      TBD |      TBD |      TBD |
| MLP + Dropout |     Yes |      TBD |      TBD |      TBD |

### CNN

| Experiment      | BatchNorm | Accuracy | F1-macro | F1-micro |
| --------------- | --------: | -------: | -------: | -------: |
| CNN baseline    |        No |      TBD |      TBD |      TBD |
| CNN + BatchNorm |       Yes |      TBD |      TBD |      TBD |

### Data Augmentation

| Experiment           | Augmentation Type | Accuracy | F1-macro | F1-micro |
| -------------------- | ----------------- | -------: | -------: | -------: |
| CNN baseline         | None              |      TBD |      TBD |      TBD |
| CNN + geometric aug. | Geometric         |      TBD |      TBD |      TBD |
| CNN + color aug.     | Color             |      TBD |      TBD |      TBD |
| CNN + full aug.      | Geometric + Color |      TBD |      TBD |      TBD |

### Class Imbalance

| Experiment              | Technique              | Accuracy | F1-macro | F1-micro |
| ----------------------- | ---------------------- | -------: | -------: | -------: |
| CNN baseline            | None                   |      TBD |      TBD |      TBD |
| CNN + weighted loss     | Weighted Cross Entropy |      TBD |      TBD |      TBD |
| CNN + balanced sampling | WeightedRandomSampler  |      TBD |      TBD |      TBD |

### Fine-Tuning

| Experiment               | Backbone                | Warmup | Accuracy | F1-macro | F1-micro |
| ------------------------ | ----------------------- | -----: | -------: | -------: | -------: |
| Frozen backbone          | ResNet-18 / MobileNetV2 |     No |      TBD |      TBD |      TBD |
| Frozen backbone + warmup | ResNet-18 / MobileNetV2 |    Yes |      TBD |      TBD |      TBD |


### Skin Lesion Classification

| Model                              | Accuracy | F1-macro | F1-micro |
| ---------------------------------- | -------: | -------: | -------: |
| MLP                                |      TBD |      TBD |      TBD |
| CNN                                |      TBD |      TBD |      TBD |
| CNN + Augmentation                 |      TBD |      TBD |      TBD |
| Fine-tuned ResNet-18 / MobileNetV2 |      TBD |      TBD |      TBD |

### Facial Emotion Recognition

| Model                              | Accuracy | F1-macro | F1-micro |
| ---------------------------------- | -------: | -------: | -------: |
| MLP                                |      TBD |      TBD |      TBD |
| CNN                                |      TBD |      TBD |      TBD |
| CNN + Augmentation                 |      TBD |      TBD |      TBD |
| Fine-tuned ResNet-18 / MobileNetV2 |      TBD |      TBD |      TBD |

## What I Learned

Through this project, I improved my understanding of:

* building a complete PyTorch training pipeline
* implementing custom datasets and dataloaders
* training MLP and CNN architectures
* applying image transformations and augmentations
* using Batch Normalization and Dropout
* dealing with imbalanced datasets
* evaluating models with F1-score and confusion matrices
* using TensorBoard for experiment tracking
* applying transfer learning with pretrained models
* comparing models through ablation studies
* analyzing overfitting and underfitting from training curves

## Author

**Mihai-Ionuț Păunescu**
