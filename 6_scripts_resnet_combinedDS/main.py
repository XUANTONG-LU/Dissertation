from eyeSurgeryDataset import EyeSurgeryDataset
from utils import TrainTransforms, EvalTransforms
from utils import set_seed
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import datetime
from utils import train, validate, iou, lr_toolsbox, loss_toolsbox, evaluate
import segmentation_models_pytorch as smp


def main(argc, writer):
    # set seed for fair comparison when doing hyperparameter tunning:
    set_seed(42)

    # step 6: log file to store all the hyperparameter in current training process:
    # use time as an unique name to the log file:
    time8training = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file_dir = "logs/record_{}.txt".format(time8training)
    
    # step 1 : instantiate dataset:
    img_width = 480
    img_height = 288
    trian_ds_1 = EyeSurgeryDataset(root=argc.ds_dir_1, subset="training",
                                    transforms=TrainTransforms(img_width=img_width, img_height=img_height), removeVacantImage=False)
    val_ds_1 = EyeSurgeryDataset(root=argc.ds_dir_1, subset="validation", 
                                 transforms=EvalTransforms(img_width=img_width, img_height=img_height), removeVacantImage=False)
    trian_ds_2 = EyeSurgeryDataset(root=argc.ds_dir_2, subset="training", num2train=argc.ds_2_num,
                                   transforms=TrainTransforms(img_width=img_width, img_height=img_height), removeVacantImage=False)
    val_ds_2 = EyeSurgeryDataset(root=argc.ds_dir_2, subset="validation", 
                                 transforms=EvalTransforms(img_width=img_width, img_height=img_height), removeVacantImage=False)
    
    test_ds = EyeSurgeryDataset(root="retinalSynthesisDataset", subset="testing",
                                transforms=EvalTransforms(img_width=img_width, img_height=img_height), removeVacantImage=False)
    
    # step 2: instantiate dataloader:
    train_dl_1 = DataLoader(dataset=trian_ds_1, batch_size=argc.batch_size, pin_memory=True, shuffle=True, num_workers=6)
    val_dl_1 = DataLoader(dataset=val_ds_1, batch_size=1, pin_memory=True, shuffle=False, num_workers=6)
    val_dl_2 = DataLoader(dataset=val_ds_2, batch_size=1, pin_memory=True, shuffle=False, num_workers=6)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, pin_memory=True, shuffle=False, num_workers=6)

    # write info about the dataset to the log file:
    with open(log_file_dir, "a") as f:
        print(f"Numer of samples in train set 1: {len(train_dl_1.dataset)}\n")
        print(f"Numer of samples in validation set 1: {len(val_dl_1.dataset)}\n")
        print(f"Numer of samples in train set 2: {len(trian_ds_2)}\n")
        print(f"Numer of samples in validation set 2: {len(val_dl_2.dataset)}\n")

    # step 3: instantiate model
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None, in_channels=3, classes=1)
    model = model.to(device=argc.device)

    # step 4: instantiate optimizer:
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=argc.lr, momentum=argc.momentum, weight_decay=argc.weight_decay
    )

    # step 6: instantiate a lr scheduler:
    lr_scheduler = lr_toolsbox(optimizer=optimizer,
                               num_step=len(train_dl_1),
                               epochs=argc.epochs)[argc.lr_scheduler]       

    # step 5: define loss function:
    # pos_weight=torch.tensor([argc.pos_weight]).to(argc.device)
    loss_fn = loss_toolsbox(pos_weight = argc.pos_weight,
                            alpha=argc.alpha,
                            gamma=argc.gamma)[argc.loss_fn]

    # step 6: log file to store all the hyperparameter in current training process:
    with open(log_file_dir, "a") as f:
        # record all the hyperparameters in the current training process:
        f.write(f"Project: Resnet-UNet on combined syntesis and cartaract dataset")
        f.write(f"Batch size: {argc.batch_size}\n")
        f.write(f"Epoch amount: {argc.epochs}\n")
        f.write(f"Max learning rate: {argc.lr}\n")
        f.write(f"Device: {argc.device}\n")
        f.write(f"Momentum: {argc.momentum}\n")
        f.write(f"Wegiht decay: {argc.weight_decay}\n")
        f.write(f"Loss function: {argc.loss_fn}\n")
        if argc.loss_fn == "focal":
            f.write(f"focal hyper alpha: {argc.alpha}\n")
            f.write(f"focal hyper gamma: {argc.gamma}\n")
        elif argc.loss_fn == "weight_bce":
            f.write(f"Positive weight: {argc.pos_weight}\n")
        f.write(f"Learning rate scheduler: {argc.lr_scheduler}\n\n")
        f.write("=" * 30 + '\n')
        f.write("="*5 + " Training Process:  " + "="*5 + "\n")
        f.write("=" * 30 + '\n')
        f.close()

    # step 7: train begin
    best_acc = -1
    f = open(log_file_dir, 'a')
    for epoch_i in range(argc.epochs):
        train_loss, train_iou = train(model=model,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      iou_fn=iou,
                                      data_loader_1 = train_dl_1,
                                      dataset_2 = trian_ds_2,
                                      device=argc.device,
                                      activation_fn=torch.nn.Sigmoid(),
                                      writer=writer,
                                      epoch_id=epoch_i, 
                                      lr_scheduler=lr_scheduler,
                                      weight = argc.ds_2_weight)
        
        f.write(f"[Train] - Epoch: {epoch_i+1}/{argc.epochs} Loss: {train_loss} Accuracy(iou): {train_iou}\n")
        print(f"[Train] - Epoch: {epoch_i+1}/{argc.epochs} Loss: {train_loss} Accuracy(iou): {train_iou}")

        val_loss_1, val_iou_1, val_loss_2, val_iou_2, mean_iou = validate(model=model,
                                                                          loss_fn=loss_fn,
                                                                          data_loader_1=val_dl_1,
                                                                          data_loader_2=val_dl_2,
                                                                          device=argc.device,
                                                                          activation_fn=torch.nn.Sigmoid(),
                                                                          iou_fn=iou,
                                                                          writer=writer,
                                                                          epoch_id=epoch_i)


        f.write(f"[Validate] - Epoch: [{epoch_i+1}/{argc.epochs}] Loss: {val_loss_1} Accuracy(iou): {val_iou_1}\n")
        print(f"[Validate] - Epoch: [{epoch_i+1}/{argc.epochs}] Loss: {val_loss_1} Accuracy(iou): {val_iou_1}")
        f.write(f"[Validate] - Epoch: [{epoch_i+1}/{argc.epochs}] Loss: {val_loss_2} Accuracy(iou): {val_iou_2}\n")
        print(f"[Validate] - Epoch: [{epoch_i+1}/{argc.epochs}] Loss: {val_loss_2} Accuracy(iou): {val_iou_2}")

        mean_loss, mean_acc, images, targets, predictions = evaluate(      
            model=model,
            data_loader=test_dl,
            device="cuda",
            activation_fn=torch.nn.Sigmoid(),
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            iou_fn=iou,)
        
        f.write(f"[Test] - Epoch: [{epoch_i+1}/{argc.epochs}] Loss: {mean_loss} Accuracy(iou): {mean_acc}\n")
        print(f"[Test] - Epoch: [{epoch_i+1}/{argc.epochs}] Loss: {mean_loss} Accuracy(iou): {mean_acc}")        

        # write to tensorboard batch-wise:
        writer.add_scalar('LossPerEpoch/Training', train_loss, epoch_i)
        writer.add_scalar('IoUPerEpoch/Training', train_iou, epoch_i)
        writer.add_scalar('LossPerEpoch/Validation_1', val_loss_1, epoch_i)
        writer.add_scalar('IoUPerEpoch/Validation_1', val_iou_1, epoch_i)
        writer.add_scalar('LossPerEpoch/Validation_2', val_loss_2, epoch_i)
        writer.add_scalar('IoUPerEpoch/Validation_2', val_iou_2, epoch_i)
        writer.add_scalar('IoUPerEpoch/Test', mean_acc, epoch_i)
        writer.add_scalar('IoUPerEpoch/Validation_all', mean_iou, epoch_i)


        # save the best weight
        if mean_acc > best_acc:
            best_acc = mean_acc
            print(f"[Save] - Model save on epoch {epoch_i+1}")
            f.write(f"[Save] - Model save on epoch {epoch_i+1} \n")
            torch.save(model.state_dict(), 'save_weights/{}.pth'.format(time8training))
        
    f.close()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument('--lr', type=float, help='initial learning rate', required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr_scheduler", type=str, required=True) # cyclic, std, constant
    parser.add_argument("--loss_fn", type=str, required=True) 
    parser.add_argument("--ds_dir_1", default= "cataractDataset", type=str, required=True) 
    parser.add_argument("--ds_dir_2", default= "retinalSynthesisDataset", type=str, required=True) 
    parser.add_argument("--ds_2_weight", default= 0.5, type=float, required=True) 
    parser.add_argument("--ds_2_num", default= 0, type=int, required=True) 

    
    # dice, focal, bce, lov, weighted_bce
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--device", default="cuda", help="training device")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

    writer = SummaryWriter()
    
    main(args, writer = writer)

    writer.flush()
    writer.close()
    
