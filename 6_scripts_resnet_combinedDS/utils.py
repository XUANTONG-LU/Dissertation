import transforms as T
import random
import torch
import numpy as np
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss, LovaszLoss
from pytorch_toolbelt import losses as L
# transformation for train samples:
class TrainTransforms:
    def __init__(self, img_width = 480, img_height = 270, degRange = 90, rotate_prob = 0.5, hflip_prob=0.5, vflip_prob=0.5):
        trans_list = [T.Resize(img_width=img_width, img_height=img_height)]
        
        if rotate_prob:
            trans_list.append(T.RandomRotation(degRange=degRange, rotate_prob=rotate_prob))
        if hflip_prob > 0:
            trans_list.append(T.RandomHorizontalFlip(flip_prob=hflip_prob))
        if vflip_prob > 0:
            trans_list.append(T.RandomVerticalFlip(flip_prob=vflip_prob))
        
        trans_list.extend([
            T.ToTensor(),
        ])
        self.transforms = T.Compose(trans_list)

    def __call__(self, img, target):
        return self.transforms(img, target)


# transformation for validation and test samples:
class EvalTransforms:
    def __init__(self, img_width = 480, img_height = 270):
        self.transforms = T.Compose([
            T.Resize(img_width=img_width, img_height=img_height),
            T.ToTensor(),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
    

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True


# sklearn version
from sklearn.metrics import confusion_matrix
def iou(pred, target):
    if torch.count_nonzero(target).item() == 0.0:
        return 1
    
    pred = pred.flatten().numpy()
    target = target.flatten().numpy()
    
    Cmat = confusion_matrix(target, pred)
    tp = Cmat[1][1]
    fp = Cmat[0][1]
    fn = Cmat[1][0]

    iou = tp / (fp + tp + fn)

    return iou


def train(model, optimizer, loss_fn, iou_fn, data_loader_1, dataset_2, device, activation_fn, writer, epoch_id, lr_scheduler, weight):
    # log_file = open(log_file_dir, "a")
    model.train()
    train_loss, train_accuracy = 0.0, 0.0
    # img_num_in_ds = len(data_loader.dataset)
    iteration_id = epoch_id * len(data_loader_1.dataset)

    for batch_id, (X_1, y_1) in enumerate(data_loader_1):
        # data from the first dataset:
        X_1, y_1 = X_1.to(device), y_1.to(device).long()

        # data from the second dataset:
        X_2 = torch.zeros_like(X_1)
        y_2 = torch.zeros_like(y_1)
        for i in range(len(X_1)):
            id = random.randint(0, len(dataset_2)-1)
            X_2[i], y_2[i] = dataset_2[id]
        X_2.to(device)
        y_2.to(device).long()

        # train:
        optimizer.zero_grad()
        outputs_1 = model(X_1)
        outputs_2 = model(X_2)
        loss = (1-weight) * loss_fn(outputs_1, y_1.float()) + weight * loss_fn(outputs_2, y_2.float())
        
        # backpropagation:
        loss.backward()
        optimizer.step()
        
        # updata lr:
        lr_scheduler.step()
        train_loss += loss.item() * X_1.size(0)

        # accmulate the loss and iou across whole epoch 
        all_outputs = torch.concatenate((outputs_1, outputs_2), axis=0)
        probability = activation_fn(all_outputs.detach().cpu())
        preds = (probability > 0.5).float()
        all_y = torch.concatenate((y_1, y_2), dim=0)
        mean_iou = iou_fn(preds, all_y.cpu())
        train_accuracy +=  mean_iou * X_1.size(0)
        
        # write to tensorboard batch-wise:
        iteration_id += X_1.size(0)
        writer.add_scalar('TrainProcess/loss', loss.item(), iteration_id)
        writer.add_scalar('TrainProcess/IoU', mean_iou, iteration_id)
        writer.add_scalar('TrainProcess/learning_rate', optimizer.param_groups[0]["lr"], iteration_id)

        # writer.add_scalar('IoU/Train', mean_iou, iteration_id)
    # return mean loss and iou per epoch:
    return (train_loss/len(data_loader_1.dataset)), (train_accuracy/len(data_loader_1.dataset)).item()
  

def validate(model, 
             loss_fn, 
             data_loader_1,
             data_loader_2,
             device,
             activation_fn,
             iou_fn,
             writer,
             epoch_id):
    model.eval()

    # for dataset 1:
    validation_loss_1, validation_accuracy_1 = 0.0, 0.0
    for batch_id, (X, y) in enumerate(data_loader_1):
        with torch.no_grad():
            X, y = X.to(device), y.to(device).long()
            outputs = model(X)
            loss = loss_fn(outputs, y.float())
            validation_loss_1 += loss.item()*X.size(0)
            probability = activation_fn(outputs.detach().cpu())
            preds = (probability > 0.5).float()
            mean_iou = iou_fn(preds, y.cpu())
            validation_accuracy_1 += mean_iou*X.size(0)

    loss_1, iou_1 = (validation_loss_1/len(data_loader_1.dataset)), (validation_accuracy_1/len(data_loader_1.dataset)).item()

    # for dataset 2:
    validation_loss_2, validation_accuracy_2 = 0.0, 0.0
    for batch_id, (X, y) in enumerate(data_loader_2):
        with torch.no_grad():
            X, y = X.to(device), y.to(device).long()
            outputs = model(X)
            loss = loss_fn(outputs, y.float())
            validation_loss_2 += loss.item()*X.size(0)
            probability = activation_fn(outputs.detach().cpu())
            preds = (probability > 0.5).float()
            mean_iou = iou_fn(preds, y.cpu())
            validation_accuracy_2 += mean_iou*X.size(0)

    loss_2, iou_2 = (validation_loss_2/len(data_loader_2.dataset)), (validation_accuracy_2/len(data_loader_2.dataset)).item()

    mean_all_iou = (validation_accuracy_1 + validation_accuracy_2) / (len(data_loader_1.dataset) + len(data_loader_2.dataset))

    return loss_1, iou_1, loss_2, iou_2, mean_all_iou.item()


def lr_toolsbox(optimizer, num_step: int, epochs: int, set_lr=0.01):
    '''here, set_lr is for the cyclic learning rate, can be change to read from args'''
    tools = {
        "std": lr_std(optimizer=optimizer,
                      num_step=num_step,
                      epochs=epochs),
        "constant": torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                      lr_lambda=lambda x: 1),
        "cyclic": torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                    base_lr=0.5*set_lr, max_lr=set_lr),
        "step": torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20*num_step, gamma=0.5),
    }

    return tools


def loss_toolsbox(pos_weight, alpha, gamma):
    tools = {"bce": torch.nn.BCEWithLogitsLoss(),
             "focal": FocalLoss(mode="binary", alpha=alpha, gamma=gamma),
             "dice": DiceLoss(mode="binary"),
             "lov": LovaszLoss(mode="binary"),
             "weight_bce": torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight])).cuda()}
    return tools


def lr_std(optimizer, 
           num_step: int,
           epochs: int,
           warmup=True,
           warmup_epochs=1,
           warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子,
        注意在训练开始之前, pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def evaluate(model, data_loader, device, activation_fn, loss_fn, iou_fn):
    model.eval()
    ytrue, ypred, xtrue = [], [], []
    test_loss, test_acc = 0.0, 0.0
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device).long()
            outputs = model(X)
            loss = loss_fn(outputs, y.float())
            probability = activation_fn(outputs.detach().cpu())
            preds = (probability > 0.5).float()
            mean_iou = iou_fn(preds, y.cpu())
            test_acc += mean_iou*X.size(0)
            test_loss += loss.item()*X.size(0)

            ytrue.append(y.detach().cpu().numpy())
            ypred.append(preds.cpu().numpy())
            xtrue.append(X.detach().cpu().numpy())
            
            
    return (test_loss/len(data_loader.dataset)), (test_acc/len(data_loader.dataset)).item(), np.concatenate(xtrue, 0), np.concatenate(ytrue, 0),  np.concatenate(ypred, 0)





