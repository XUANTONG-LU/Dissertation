# clearn the folder test_result and reconstruct the folder:
import os
import shutil
from eyeSurgeryDataset import EyeSurgeryDataset
from utils import TrainTransforms, EvalTransforms, set_seed
from utils import evaluate, iou
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from utils import EvalTransforms
import matplotlib.pyplot as plt
import numpy as np
import random

def prepare_folder():
    if not os.path.exists("test_result"):
        os.mkdir("test_result")
        os.makedirs("test_result/images")
        os.makedirs("test_result/annotations")
        os.makedirs("test_result/predictions")
    else:
        shutil.rmtree("test_result")
        os.mkdir("test_result")
        os.makedirs("test_result/images")
        os.makedirs("test_result/annotations")
        os.makedirs("test_result/predictions")

'''
This script do prediction on test set. given the weight that you want to test, the 
programme will return the IoU and a figure that indicating the real performance of the model.
'''

def test_plot(args):
    # step 1: instantiate a model
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None, in_channels=3, classes=1)
    model = model.to(device="cuda")

    # step 2: load weight
    weight_file = str(args.weight).split('/')[-1]
    print(f"[Testing] - check the weight {weight_file}")
    model.load_state_dict(torch.load("save_weights/" + weight_file))
    print(f"[Testing] - Loading the weight {weight_file} DONE!")

    # step 3: instantiate a dataset
    test_ds = EyeSurgeryDataset(root = "retinalSynthesisDataset", subset='testing', transforms=EvalTransforms(img_width=480, img_height=288), removeVacantImage=False)

    # step 4: instantiate a dataloader
    test_dl = DataLoader(dataset=test_ds, batch_size=1, pin_memory=True, shuffle=False, num_workers=10)
    print(f"[Testing] - Loading dataset DONE!")

    # step 5: evaluation fn return mean loss/mean acc/images/ true mask/prediction:
    mean_loss, mean_acc, images, targets, predictions = evaluate(
        model=model,
        data_loader=test_dl,
        device="cuda",
        activation_fn=torch.nn.Sigmoid(),
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        iou_fn=iou,)

    print(f"[Testing] - Loss: {mean_loss} IoU: {mean_acc}")

    # step 6: plot the result
    for i in range(len(test_ds)):
        save_img = np.moveaxis(images[i], 0, -1)
        plt.imsave(f"test_result/images/{str(i).zfill(3)}.png", save_img)

        save_mask = np.moveaxis(targets[i], 0, -1).squeeze()
        plt.imsave(f"test_result/annotations/{str(i).zfill(3)}.png", save_mask, cmap='gray')

        save_pred = np.moveaxis(predictions[i], 0, -1).squeeze()
        plt.imsave(f"test_result/predictions/{str(i).zfill(3)}.png", save_pred, cmap='gray')




def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet testing")
    parser.add_argument("--weight", default='save_weights/20230901-140659.pth', help="weight_file")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    import os
    
    args = parse_args()
    
    prepare_folder()
    
    test_plot(args)