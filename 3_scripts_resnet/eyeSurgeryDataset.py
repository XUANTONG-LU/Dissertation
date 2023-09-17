from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
import cv2 as cv

def binarize_pixel_fn(pixel_i):
    return 255 if pixel_i == 7 else 0


class EyeSurgeryDataset(Dataset):
    def __init__(self, root: str, subset: str, transforms=None, removeVacantImage=True):
        super(EyeSurgeryDataset, self).__init__()
        # subset define which dataset you use, can be only choose from ["training", "testing", "validation"]
        data_root = os.path.join(root, subset)

        # get all img id:
        all_img_names = [i for i in os.listdir(os.path.join(data_root, "Images")) if i.endswith(".png")] 
        all_img_names.sort()

        if removeVacantImage:
            # get all img id which are all black mask:
            img_names_with_no_tools = os.listdir("noToolsImages")
            # remove img id that are all black mask:
            img_names = [x for x in all_img_names if x not in img_names_with_no_tools]
        else:
            img_names = all_img_names

        # get all imges and mask path (relative path to curret directorey)
        self.img_list = [os.path.join(data_root, "Images", i) for i in img_names] # all images full directory
        self.mask_list = [os.path.join(data_root, "Annotations", i) for i in img_names]
        
        # binarize mask function:
        self.binarize_mask_fn = np.vectorize(binarize_pixel_fn)

        # transform:
        self.transforms = transforms

    def __getitem__(self, idx):
        # step 1: PIL Image, property 'size' return the h x w of image, channel should be check after converting to numpy array:
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask_list[idx]).convert('L')

        # step 2: change mask to np array and change pixels equal to 7 to 255 and else to 0:
        mask_np = np.array(mask)
        binary_mask_np = self.binarize_mask_fn(mask_np)
        # convert back to PIL:
        mask = Image.fromarray(np.uint8(binary_mask_np)) # 0 and 255 'L' Image

        # step 3: transform
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)
    

if __name__ == "__main__":
    from utils import TrainTransforms, EvalTransforms, set_seed
    import matplotlib.pyplot as plt
    import random
    set_seed(42)

    train_ds = EyeSurgeryDataset(root = "retinalWithLabelDataset", subset='training', transforms=TrainTransforms(img_width=480, img_height=288), removeVacantImage=True)
    fig, axs = plt.subplots(nrows=4, ncols=2, squeeze=False, figsize=(10,20))
    idx = [random.randint(0, len(train_ds)-1) for _ in range(4)]

    for img_i, i in enumerate(idx):
        item = train_ds[i]
        img, mask = item
        ax_img = axs[img_i, 0]
        ax_img.axis('off')
        ax_img.imshow(np.moveaxis(np.asarray(img), 0, -1))
        ax_mask = axs[img_i, 1]
        ax_mask.axis('off')
        ax_mask.imshow(np.moveaxis(np.asarray(mask), 0, -1), cmap='gray')

        if img_i == 3:
            print("The pixel range of image: %.2f ~ %.2f" % (img.max().tolist(), img.min().tolist()))
            print("The unique pixel value of mask is: ", torch.unique(mask))
            print("The dtype of image and mask: ", img.dtype, mask.dtype)
            break
    plt.show()
