import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from models import CI_model
from utils import *

set_seed(42)

model_path = save_metrics(f"save_model")
current_acc = 0
best_epoch = 0

# Set the device to GPU 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
id_device = 0
device = device = f"cuda:{id_device}" if torch.cuda.is_available() else "cpu"
# if colab is used, set the device to GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
num_test = 4
letter = "J" # letter from the person who is running the code
num_epochs = 100
batch_size = 2**7
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
milestones = [30, 50, 70, 80]
gamma = 0.1
SPC_portion = 0.2 # 1.0 for 100% CAP

channels, im_size, num_classes, class_names, _, _, testloader, trainloader, valoader = get_dataset("CIFAR10", "data", batch_size, 42)
im_size = (channels, im_size[-2], im_size[-1])

model = CI_model(input_size=im_size,
        snapshots=int(SPC_portion * 32 * 32),
        real="False").to(device)

# Loss and regularization
CE_LOSS = nn.CrossEntropyLoss()
accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
wandb.init(project="Classification_CIFAR10_Acc_vs_SPC",name=f"Prueba {letter}{num_test} - 20% CAP _lr: {lr} _momentum: {momentum} _weight_decay: {weight_decay} _milestones: {milestones} _gamma: {gamma}",config={"num_epochs": num_epochs})

for epoch in range(num_epochs):
  model.train()

  train_loss = AverageMeter()
  train_acc = AverageMeter()

  data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="blue")
  for _, train_data in data_loop_train:
    x_imgs, x_labels = train_data
    x_imgs = x_imgs.to(device)
    x_labels = x_labels.to(device)
    x_hat = model(x_imgs)
    pred_labels = torch.argmax(x_hat, dim=1)
    loss_train = CE_LOSS(x_hat, x_labels)
    acc_train = accuracy(pred_labels, x_labels)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    train_loss.update(loss_train.item())
    train_acc.update(acc_train.item())
    data_loop_train.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_train.set_postfix(loss=train_loss.avg, acc=train_acc.avg)
    
  
  with torch.no_grad():
    model.eval()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour="green")
    for _, val_data in data_loop_val:
      x_imgs, x_labels = val_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)
      x_hat = model(x_imgs)
      pred_labels = torch.argmax(x_hat, dim=1)
      loss_val = CE_LOSS(x_hat, x_labels)
      acc_val = accuracy(pred_labels, x_labels)
      val_loss.update(loss_val.item())
      val_acc.update(acc_val.item())
      data_loop_val.set_description(f"Epoch: {epoch+1}/{num_epochs}")
      data_loop_val.set_postfix(loss=val_loss.avg, acc=val_acc.avg)
  
  scheduler.step()
  if val_acc.avg > current_acc:
            current_acc = val_acc.avg
            print(f"Saving model with Accuracy: {current_acc}")
            torch.save(model.state_dict(), f"{model_path}/model.pth")
            best_epoch = epoch

  wandb.log({"train_loss": train_loss.avg,
              "val_loss": val_loss.avg,
              "train_acc": train_acc.avg,
              "val_acc": val_acc.avg,})

test_loss = AverageMeter()
test_acc = AverageMeter()

del model

model = CI_model(input_size=im_size,
        snapshots=int(SPC_portion * 32 * 32),
        real="False").to(device)

model.load_state_dict(torch.load(f"{model_path}/model.pth"))

data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="magenta")
with torch.no_grad():
  model.eval()
  for _, test_data in data_loop_test:
    x_imgs, x_labels = test_data
    x_imgs = x_imgs.to(device)
    x_labels = x_labels.to(device)
    x_hat = model(x_imgs)
    pred_labels = torch.argmax(x_hat, dim=1)
    loss_test = CE_LOSS(x_hat, x_labels)
    acc_test = accuracy(pred_labels, x_labels)
    test_loss.update(loss_test.item())
    test_acc.update(acc_test.item())
    data_loop_test.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_test.set_postfix(loss=test_loss.avg, acc=test_acc.avg)

wandb.log({"test_loss": test_loss.avg,
            "test_acc": test_acc.avg,
            "best_epoch": best_epoch})

wandb.finish()