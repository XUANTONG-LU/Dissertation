# Dissertation
This repository contains all the Python scripts required for the training of an image segmentation model for retinal surgery. The model is implemented by Pytorch.
For the training, the dataset should be a floder as the following structure:
```
Dataset_folder:
|_training
  |_Images
  |_Annotations
|_testing
  |_Images
  |_Annotations
|_validation
  |_Images
  |_Annotations
```
Each folder in the model corresponds to a training method mentioned in the dissertation. To train a model, simply run the 'main.py' file in each folder.
