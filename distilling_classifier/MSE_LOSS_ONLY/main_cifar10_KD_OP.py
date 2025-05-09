import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import Accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from models import CI_model
from utils import *

def main(args):
  set_seed(args.seed)

  path_name = f"lr_{args.lr}_b_{args.batch_size}_e_{args.num_epochs}_momentum_{args.momentum}_wd_{args.weight_decay}_milestone_{args.milestones}_gamma_{args.gamma}_snap_t_{int(args.SPC_portion_tchr*100)}_snap_s_{int(args.SPC_portion_st*100)}_ds_{args.dataset}_sd_{args.seed}_l1_{args.lambda1}_l2_{args.lambda2}_dropout_{args.dropout}"


  args.save_path = args.save_path + path_name

  images_path, model_path = save_metrics(f"args.save_path")
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
  MSE = nn.MSELoss()
  accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

  student = CI_model(input_size=im_size,
          snapshots=int(args.SPC_portion_st * 32 * 32),
          real=args.real_st, dropout_rate=args.dropout).to(device) # True for real, False for binary

  teacher = CI_model(input_size=im_size,
          snapshots=int(args.SPC_portion_tchr * 32 * 32),
          real=args.real_tchr, dropout_rate=args.dropout).to(device) # True for real, False for binary

  teacher.load_state_dict(torch.load(args.teacher_path)) # cada run con sus pesos

  for param in teacher.parameters():
      param.requires_grad = False

  teacher.eval()
  optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

  wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
  wandb.init(project=args.project_name, name="KD_mse_" + path_name, config=args)

  for epoch in range(args.num_epochs):
    student.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    train_mse_loss = AverageMeter()
    train_deco_loss = AverageMeter()

    data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="blue")
    for _, train_data in data_loop_train:
      x_imgs, x_labels = train_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)

      ys_train, x_hat_s_train, resnet_features_s_train, HTt = student(x_imgs)
      yt_train, x_hat_t_train, resnet_features_t_train, HTs = teacher(x_imgs)

      pred_labels_s = torch.argmax(x_hat_s_train, dim=1)

      loss_deco = CE_LOSS(x_hat_s_train, x_labels)

      loss_mse = MSE(HTt, HTs)

      loss_train = (args.lambda1*loss_deco + args.lambda2*loss_mse)

      optimizer.zero_grad()
      loss_train.backward()
      optimizer.step()

      train_loss.update(loss_train.item())
      train_deco_loss.update(loss_deco.item())
      train_mse_loss.update(loss_mse.item())
      train_acc.update(accuracy(pred_labels_s, x_labels).item())
      data_loop_train.set_description(f"Epoch: {epoch+1}/{args.num_epochs}")
      data_loop_train.set_postfix(loss=train_loss.avg, acc=train_acc.avg)

    with torch.no_grad():
      student.eval()

      val_loss = AverageMeter()
      val_acc = AverageMeter()

      val_mse_loss = AverageMeter()
      val_deco_loss = AverageMeter()

      data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour="green")
      for _, val_data in data_loop_val:
        x_imgs, x_labels = val_data
        x_imgs = x_imgs.to(device)
        x_labels = x_labels.to(device)

        ys_val, x_hat_s_val, resnet_features_s_val, HTt  = student(x_imgs)
        yt_val, x_hat_t_val, resnet_features_t_val, HTs = teacher(x_imgs)

        pred_labels_s = torch.argmax(x_hat_s_val, dim=1)
        loss_deco = CE_LOSS(x_hat_s_val, x_labels)

        loss_mse = MSE(HTt, HTs)

        loss_val = (args.lambda1*loss_deco + args.lambda2*loss_mse)

        val_loss.update(loss_val.item())
        val_deco_loss.update(loss_deco.item())
        val_mse_loss.update(loss_mse.item())
        val_acc.update(accuracy(pred_labels_s, x_labels).item())
        data_loop_val.set_description(f"Epoch: {epoch+1}/{args.num_epochs}")
        data_loop_val.set_postfix(loss=val_loss.avg, acc=val_acc.avg)

    scheduler.step()
    if val_acc.avg > current_acc:
              current_acc = val_acc.avg
              print(f"Saving model with Accuracy: {current_acc}")
              torch.save(student.state_dict(), f"{model_path}/model.pth")
              best_epoch = epoch

    image_array = save_coded_apertures(
       student.system_layer, 8, 2, images_path, f"coded_aperture_{epoch}"
       )
    images = wandb.Image(image_array, caption=f"Epoch: {epoch}")

    wandb.log({"train_loss": train_loss.avg,
                "val_loss": val_loss.avg,
                "train_acc": train_acc.avg,
                "val_acc": val_acc.avg,
                "train_mse_loss": train_mse_loss.avg,
                "train_deco_loss": train_deco_loss.avg,
                "val_mse_loss": val_mse_loss.avg,
                "val_deco_loss": val_deco_loss.avg,
                "coded_aperture": images if epoch % 20 == 0 else None,
                "logits_s_train": wandb.Histogram(x_hat_s_train[0].detach().cpu().numpy(), num_bins=10) if epoch % 10 == 0 else None,
                "logits_t_train": wandb.Histogram(x_hat_t_train[0].detach().cpu().numpy(), num_bins=10) if epoch % 10 == 0 else None,
                "logits_s_val": wandb.Histogram(x_hat_s_val[0].detach().cpu().numpy(), num_bins=10) if epoch % 10 == 0 else None,
                "logits_t_val": wandb.Histogram(x_hat_t_val[0].detach().cpu().numpy(), num_bins=10) if epoch % 10 == 0 else None})


  test_loss = AverageMeter()
  test_acc = AverageMeter()

  test_mse_loss = AverageMeter()
  test_deco_loss = AverageMeter()

  del student

  student = CI_model(input_size=im_size,
          snapshots=int(args.SPC_portion_st * 32 * 32),
          real=args.real_st, dropout_rate=args.dropout).to(device) # True for real, False for binary

  student.load_state_dict(torch.load(f"{model_path}/model.pth"))

  data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="magenta")
  with torch.no_grad():
    student.eval()
    for _, test_data in data_loop_test:
      x_imgs, x_labels = test_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)

      ys_test, x_hat_s_test, resnet_features_s_test, HTt = student(x_imgs)
      yt_test, x_hat_t_test, resnet_features_t_test, HTs = teacher(x_imgs)

      pred_labels_s = torch.argmax(x_hat_s_test, dim=1)
      loss_deco = CE_LOSS(x_hat_s_test, x_labels)

      loss_mse = MSE(HTt, HTs)

      loss_test = (args.lambda1*loss_deco + args.lambda2*loss_mse)
      test_loss.update(loss_test.item())
      test_deco_loss.update(loss_deco.item())
      test_mse_loss.update(loss_mse.item())
      test_acc.update(accuracy(pred_labels_s, x_labels).item())
      data_loop_test.set_description(f"Epoch: {epoch+1}/{args.num_epochs}")
      data_loop_test.set_postfix(loss=test_loss.avg, acc=test_acc.avg)

  wandb.log({"test_loss": test_loss.avg,
              "test_acc": test_acc.avg,
              "best_epoch": best_epoch,
              "test_mse_loss": test_mse_loss.avg})

  wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2**7)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--milestones", nargs="+", type=int, default = [20, 40], help="Lista")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--SPC_portion_tchr", type=float, default=0.2)
    parser.add_argument("--SPC_portion_st", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.75)
    parser.add_argument(
        "--teacher_path",
        type=str,
        default=r"save_model_t\model_binary_20.pth",
    )
    parser.add_argument("--save_path", type=str, default="WEIGHTS/SPC_KD_TEST/")
    parser.add_argument("--project_name", type=str, default="KD_MSE_TEST")
    parser.add_argument("--real_st", type=str, default="False") # True for real, False for binary
    parser.add_argument("--real_tchr", type=str, default="False") # True for real, False for binary
    parser.add_argument("--dropout", type=float, default=0.4)

    os.chdir("distilling_classifier/MSE_LOSS_ONLY")
    args = parser.parse_args()
    print(args)
    main(args)