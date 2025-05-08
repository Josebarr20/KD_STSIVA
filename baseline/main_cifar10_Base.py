import argparse
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

def main(args):
  set_seed(args.seed) 

  path_name = f"lr_{args.lr}_b_{args.batch_size}_e_{args.num_epochs}_momentum_{args.momentum}_wd_{args.weight_decay}_milestone_{args.milestones}_gamma_{args.gamma}_cap_{int(args.SPC_portion*100)}_dropout_{args.dropout}_type_{args.type_t}_real_{args.real}/"

  args.save_path = args.save_path + path_name

  images_path, model_path = save_metrics(f"save_model")
  current_acc = 0
  best_epoch = 0

  torch.autograd.set_detect_anomaly(True)

  # Set the device to GPU 
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  id_device = 0
  device = f"cuda:{id_device}" if torch.cuda.is_available() else "cpu"
  # if colab is used, set the device to GPU
  # device = "cuda" if torch.cuda.is_available() else "cpu"

  channels, im_size, num_classes, class_names, _, _, testloader, trainloader, valoader = get_dataset(
     args.dataset, "data", args.batch_size, args.seed) 
  im_size = (channels, im_size[-2], im_size[-1])

  CE_LOSS = nn.CrossEntropyLoss()
  accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

  model = CI_model(input_size=im_size,
          snapshots=int(args.SPC_portion * 32 * 32),
          real=args.real, dropout_rate=args.dropout).to(device) # True for real, False for binary

  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

  wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
  wandb.init(project=args.project_name, name=f"B_{args.type_t}_" + path_name, config=args)

  for epoch in range(args.num_epochs):
    model.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="blue")
    for _, train_data in data_loop_train:
      x_imgs, x_labels = train_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)
      x_hat, resnet_features = model(x_imgs)
      pred_labels = torch.argmax(x_hat, dim=1)
      loss_train = CE_LOSS(x_hat, x_labels)
      acc_train = accuracy(pred_labels, x_labels)

      optimizer.zero_grad()
      loss_train.backward()
      optimizer.step()

      train_loss.update(loss_train.item())
      train_acc.update(acc_train.item())
      data_loop_train.set_description(f"Epoch: {epoch+1}/{args.num_epochs}")
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
        x_hat, resnet_features = model(x_imgs)
        pred_labels = torch.argmax(x_hat, dim=1)
        loss_val = CE_LOSS(x_hat, x_labels)
        acc_val = accuracy(pred_labels, x_labels)
        val_loss.update(loss_val.item())
        val_acc.update(acc_val.item())
        data_loop_val.set_description(f"Epoch: {epoch+1}/{args.num_epochs}")
        data_loop_val.set_postfix(loss=val_loss.avg, acc=val_acc.avg)
    
    scheduler.step()
    if val_acc.avg > current_acc:
              current_acc = val_acc.avg
              print(f"Saving model with Accuracy: {current_acc}")
              torch.save(model.state_dict(), f"{model_path}/model_{args.type_t}_{int(args.SPC_portion*100)}.pth")
              best_epoch = epoch
    
    image_array = save_coded_apertures(
       model.system_layer, 8, 2, images_path, f"coded_aperture_{epoch}"
       )
    images = wandb.Image(image_array, caption=f"Epoch: {epoch}")

    wandb.log({"train_loss": train_loss.avg,
                "val_loss": val_loss.avg,
                "train_acc": train_acc.avg,
                "val_acc": val_acc.avg,
                "coded_aperture": images if epoch % 10 == 0 else None})

  test_loss = AverageMeter()
  test_acc = AverageMeter()

  del model

  model = CI_model(input_size=im_size,
          snapshots=int(args.SPC_portion * 32 * 32),
          real=args.real, dropout_rate = args.dropout).to(device) # True for real, False for binary

  model.load_state_dict(torch.load(f"{model_path}/model_{args.type_t}_{int(args.SPC_portion*100)}.pth"))

  data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="magenta")
  with torch.no_grad():
    model.eval()
    for _, test_data in data_loop_test:
      x_imgs, x_labels = test_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)
      x_hat, resnet_features = model(x_imgs)
      pred_labels = torch.argmax(x_hat, dim=1)
      loss_test = CE_LOSS(x_hat, x_labels)
      acc_test = accuracy(pred_labels, x_labels)
      test_loss.update(loss_test.item())
      test_acc.update(acc_test.item())
      data_loop_test.set_description(f"Epoch: {epoch+1}/{args.num_epochs}")
      data_loop_test.set_postfix(loss=test_loss.avg, acc=test_acc.avg)

  wandb.log({"test_loss": test_loss.avg,
              "test_acc": test_acc.avg,
              "best_epoch": best_epoch})

  wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2**7)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--milestones", nargs="+", type=int, default = [30, 50, 70, 80], help="Lista")
    parser.add_argument("--gamma", type=float, default=0.1)   
    parser.add_argument("--SPC_portion", type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default="CIFAR10") 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="WEIGHTS/BASELINE_TEST/")
    parser.add_argument("--project_name", type=str, default="Classification_CIFAR10_Baseline_pilot")
    parser.add_argument("--type_t", type=str, default="real")
    parser.add_argument("--real", type=str, default="True") # True for real, False for binary
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    print(args)
    main(args)