# Dissertation
This repository contains all the Python scripts required for Final Project for COMP0132: MSc Robotics and Computation Dissertation (22/23). The model is implemented by Pytorch.
For the training, the dataset should be a folder as the following structure:
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
Each folder in the model corresponds to a training method mentioned in the dissertation. To train a model, run the `main.py` file in each folder. Once the training is finished, several folders will be automatically created containing the information about the training.
```
logs: the log of the training process containing the configuration of the training, like batch size, epoch, loss function, etc. The training loss and validation loss will be reported batch-wise.
save_weights: the weights saved after training. The training details are in the logs folder, which has a .txt file with the same name.
runs: the folder contains the history of training, which can be visualised by tensorboard.
```
For the generation of synthesis data, examples are given in the folder `synthesisGenerator`. By putting the background and tools that want to be synthesised into a simple image and then running the Python script, a batch of synthesis images will be generated.
```
synthesisGenerator:
|_backgrounds:
  |_backgrounds_1
  |_backgrounds_2
  :
  :
|_tools
  |_Annotations:
    |_1.png
    |_2.png
    :
    :
  |_Images:
    |_1.png
    |_2.png
    :
    :
|_retinalSynthesisGenerator.ipynb
```

