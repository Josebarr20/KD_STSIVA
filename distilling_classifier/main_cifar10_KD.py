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
from kd_loss import *

def main(args):
  set_seed(args.seed)

  path_name = f"lr_{args.lr}_b_{args.batch_size}_e_{args.num_epochs}_momentum_{args.momentum}_wd_{args.weight_decay}_milestone_{args.milestones}_gamma_{args.gamma}_snap_t_{int(args.SPC_portion_tchr*100)}_snap_s_{int(args.SPC_portion_st*100)}_ds_{args.dataset}_sd_{args.seed}_lossr_{args.loss_response}_T_{args.temperature}_l1_{args.lambda1}_l2_{args.lambda2}_l3_{args.lambda3}"

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
  # CORR_LOSS = Correlation(batch_size=batch_size).to(device)
  kl_div_loss = nn.KLDivLoss(log_target=True)
  accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

  student = CI_model(input_size=im_size,
          snapshots=int(args.SPC_portion_st * 32 * 32),
          real=args.real_st).to(device) # True for real, False for binary

  teacher = CI_model(input_size=im_size,
          snapshots=int(args.SPC_portion_tchr * 32 * 32),
          real=args.real_tchr).to(device) # True for real, False for binary

  teacher.load_state_dict(torch.load(args.teacher_path)) # cada run con sus pesos

  for param in teacher.parameters():
      param.requires_grad = False

  teacher.eval()
  optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

  wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
  wandb.init(project=args.project_name, name="KD_test_losses" + path_name, config=args)

  for epoch in range(args.num_epochs):
    student.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    # train_labels_loss = AverageMeter()
    train_optics_loss = AverageMeter()
    train_kl_loss = AverageMeter()
    train_deco_loss = AverageMeter()

    data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="blue")
    for _, train_data in data_loop_train:
      x_imgs, x_labels = train_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)

      x_hat_s_train, resnet_features_s_train = student(x_imgs)
      x_hat_t_train, resnet_features_t_train = teacher(x_imgs)

      pred_labels_s = torch.argmax(x_hat_s_train, dim=1)

      loss_deco = CE_LOSS(x_hat_s_train, x_labels)

      loss_optics = kd_rb_spc(
                  pred_teacher=x_hat_t_train,
                  pred_student=x_hat_s_train,
                  loss_type=args.loss_response,
                  ca_s=student.system_layer.H,
                  ca_t=teacher.system_layer.H
                  )

  #     loss_labels = CORR_LOSS(inputs=(x_hat_s, x_hat_t))

      soft_targets_train = nn.functional.log_softmax(x_hat_t_train / args.temperature, dim=-1)
      soft_prob_train = nn.functional.log_softmax(x_hat_s_train / args.temperature, dim=-1)
      loss_kl = kl_div_loss(soft_prob_train, soft_targets_train)

      loss_train = (args.lambda1*loss_deco + args.lambda2*loss_optics + args.lambda3*loss_kl)   

      optimizer.zero_grad()
      loss_train.backward()
      optimizer.step()

      train_loss.update(loss_train.item())
      train_deco_loss.update(loss_deco.item())
      train_optics_loss.update(loss_optics.item())
      train_kl_loss.update(loss_kl.item())
      # train_labels_loss.update(loss_labels.item())
      train_acc.update(accuracy(pred_labels_s, x_labels).item())
      data_loop_train.set_description(f"Epoch: {epoch+1}/{args.num_epochs}")
      data_loop_train.set_postfix(loss=train_loss.avg, acc=train_acc.avg)
    
    with torch.no_grad():
      student.eval()

      val_loss = AverageMeter()
      val_acc = AverageMeter()

      # val_labels_loss = AverageMeter()
      val_optics_loss = AverageMeter()
      val_kl_loss = AverageMeter()
      val_deco_loss = AverageMeter()

      data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour="green")
      for _, val_data in data_loop_val:
        x_imgs, x_labels = val_data
        x_imgs = x_imgs.to(device)
        x_labels = x_labels.to(device)

        x_hat_s_val, resnet_features_s_val = student(x_imgs)
        x_hat_t_val, resnet_features_t_val = teacher(x_imgs)

        pred_labels_s = torch.argmax(x_hat_s_val, dim=1)
        loss_deco = CE_LOSS(x_hat_s_val, x_labels)

        loss_optics = kd_rb_spc(
                  pred_teacher=x_hat_t_val,
                  pred_student=x_hat_s_val,
                  loss_type=args.loss_response,
                  ca_s=student.system_layer.H,
                  ca_t=teacher.system_layer.H
                  )

  #       loss_labels = CORR_LOSS(inputs=(x_hat_s, x_hat_t))

        soft_targets_val = nn.functional.log_softmax(x_hat_t_val / args.temperature, dim=-1)
        soft_prob_val = nn.functional.log_softmax(x_hat_s_val / args.temperature, dim=-1)
        loss_kl = kl_div_loss(soft_prob_val, soft_targets_val)

        loss_val = (args.lambda1*loss_deco + args.lambda2*loss_optics + args.lambda3*loss_kl)    

        val_loss.update(loss_val.item())
        val_deco_loss.update(loss_deco.item())
        val_optics_loss.update(loss_optics.item())
        val_kl_loss.update(loss_kl.item())
        # val_labels_loss.update(loss_labels.item())
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
                # "train_labels_loss": train_labels_loss.avg,
                "train_optics_loss": train_optics_loss.avg,
                "train_kl_loss": train_kl_loss.avg,
                "train_deco_loss": train_deco_loss.avg,
                # "val_labels_loss": val_labels_loss.avg,
                "val_optics_loss": val_optics_loss.avg,
                "val_kl_loss": val_kl_loss.avg,
                "val_deco_loss": val_deco_loss.avg,
                "coded_aperture": images if epoch % 20 == 0 else None,
                "logits_s_train": wandb.Histogram(x_hat_s_train.detach().cpu().numpy()) if epoch % 10 == 0 else None,
                "logits_t_train": wandb.Histogram(x_hat_t_train.detach().cpu().numpy()) if epoch % 10 == 0 else None,
                "logits_s_val": wandb.Histogram(x_hat_s_val.detach().cpu().numpy()) if epoch % 10 == 0 else None,
                "logits_t_val": wandb.Histogram(x_hat_t_val.detach().cpu().numpy()) if epoch % 10 == 0 else None,
                "probs_s_train": wandb.Histogram(soft_prob_train.detach().cpu().numpy()) if epoch % 10 == 0 else None,
                "probs_t_train": wandb.Histogram(soft_targets_train.detach().cpu().numpy()) if epoch % 10 == 0 else None,
                "probs_s_val": wandb.Histogram(soft_prob_val.detach().cpu().numpy()) if epoch % 10 == 0 else None,
                "probs_t_val": wandb.Histogram(soft_targets_val.detach().cpu().numpy()) if epoch % 10 == 0 else None})


  test_loss = AverageMeter()
  test_acc = AverageMeter()

  # test_labels_loss = AverageMeter()
  test_optics_loss = AverageMeter()
  test_kl_loss = AverageMeter()
  test_deco_loss = AverageMeter()

  del student

  student = CI_model(input_size=im_size,
          snapshots=int(args.SPC_portion_st * 32 * 32),
          real=args.real_st).to(device) # True for real, False for binary

  student.load_state_dict(torch.load(f"{model_path}/model.pth"))

  data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="magenta")
  with torch.no_grad():
    student.eval()
    for _, test_data in data_loop_test:
      x_imgs, x_labels = test_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)

      x_hat_s_test, resnet_features_s_test = student(x_imgs)
      x_hat_t_test, resnet_features_t_test = teacher(x_imgs)

      pred_labels_s = torch.argmax(x_hat_s_test, dim=1)
      loss_deco = CE_LOSS(x_hat_s_test, x_labels)

      loss_optics = kd_rb_spc(
                  pred_teacher=x_hat_t_test,
                  pred_student=x_hat_s_test,
                  loss_type=args.loss_response,
                  ca_s=student.system_layer.H,
                  ca_t=teacher.system_layer.H
                  )

  #     loss_labels = CORR_LOSS(inputs=(x_hat_s, x_hat_t))

      soft_targets_test = nn.functional.log_softmax(x_hat_t_test / args.temperature, dim=-1)
      soft_prob_test = nn.functional.log_softmax(x_hat_s_test / args.temperature, dim=-1)
      loss_kl = kl_div_loss(soft_prob_test, soft_targets_test)

      loss_test = (args.lambda1*loss_deco + args.lambda2*loss_optics + args.lambda3*loss_kl)    

      test_loss.update(loss_test.item())
      test_deco_loss.update(loss_deco.item())
      test_optics_loss.update(loss_optics.item())
      test_kl_loss.update(loss_kl.item())
      # test_labels_loss.update(loss_labels.item())
      test_acc.update(accuracy(pred_labels_s, x_labels).item())
      data_loop_test.set_description(f"Epoch: {epoch+1}/{args.num_epochs}")
      data_loop_test.set_postfix(loss=test_loss.avg, acc=test_acc.avg)

  wandb.log({"test_loss": test_loss.avg,
              "test_acc": test_acc.avg,
              "best_epoch": best_epoch,
              # "test_labels_loss": test_labels_loss.avg,
              "test_optics_loss": test_optics_loss.avg,
              "test_kl_loss": test_kl_loss.avg,
              "test_deco_loss": test_deco_loss.avg})

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
    parser.add_argument("--SPC_portion_tchr", type=float, default=0.2)
    parser.add_argument("--SPC_portion_st", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_response", type=str, default="gram")
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--lambda3", type=float, default=1.0)
    parser.add_argument(
        "--teacher_path",
        type=str,
        default=r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model_binary_10.pth",
    )
    parser.add_argument("--save_path", type=str, default="WEIGHTS/SPC_KD_TEST/")
    parser.add_argument("--project_name", type=str, default="KD_LOSSES_TEST")
    parser.add_argument("--real_st", type=str, default="False") # True for real, False for binary
    parser.add_argument("--real_tchr", type=str, default="False") # True for real, False for binary
    

    args = parser.parse_args()
    print(args)
    main(args)