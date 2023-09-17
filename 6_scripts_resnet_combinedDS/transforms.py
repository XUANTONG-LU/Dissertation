import random
import torch
import torchvision


class Compose(object):
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, image, target):
        for t in self.transforms_list:
            image, target = t(image, target)
        return image, target


class Resize(object):
    def __init__(self, img_width:int, img_height:int):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, image, target):
        image = torchvision.transforms.functional.resize(image, (self.img_height, self.img_width))
        # to reserve the mask pixel, use torchvision.transforms.InterpolationMode.NEAREST
        target = torchvision.transforms.functional.resize(target, (self.img_height, self.img_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        return image, target


class RandomRotation(object):
    def __init__(self, degRange:int, rotate_prob: float):
        self.degree = random.randint(-degRange, degRange)
        self.rotate_prob = rotate_prob

    def __call__(self, image, target):
        if random.random() < self.rotate_prob:
            image = torchvision.transforms.functional.rotate(image, angle = self.degree, fill=0)
            target = torchvision.transforms.functional.rotate(target, angle = self.degree, fill=0)
        return image, target
    

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob: float):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = torchvision.transforms.functional.hflip(image)
            target = torchvision.transforms.functional.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob:float):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = torchvision.transforms.functional.vflip(image)
            target = torchvision.transforms.functional.vflip(target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.ToTensor()(image)
        target = torchvision.transforms.ToTensor()(target)
        
        # assert torch.equal(torch.unique(target), torch.tensor([0., 1.])) or torch.equal(torch.unique(target), torch.tensor([0.]))

        return image, target
    

