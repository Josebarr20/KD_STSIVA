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

set_seed(42)

model_path = save_metrics(f"save_model")
current_acc = 0
best_epoch = 0

torch.autograd.set_detect_anomaly(True)

# Set the device to GPU 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
id_device = 0
device = device = f"cuda:{id_device}" if torch.cuda.is_available() else "cpu"
# if colab is used, set the device to GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"

# Run settings
num_test = 1
letter = "J" # letter from the person who is running the code

# Hyperparameters
num_epochs = 100
batch_size = 2**7
lr = 0.005
momentum = 0.9
weight_decay = 5e-4
milestones = [15, 50, 70, 80]
gamma = 0.1
SPC_portion_st = 0.1
SPC_portion_tchr = 0.2
lambda1 = 0.4 # CE (deco)
lambda2 = 0 # kd_rb_spc (optics)
lambda3 = 0 # correlation (labels)
lambda4 = 0 # softmax (kl)

channels, im_size, num_classes, class_names, _, _, testloader, trainloader, valoader = get_dataset("CIFAR10", "data", batch_size, 42)
im_size = (channels, im_size[-2], im_size[-1])

# Regularizers and metrics
CE_LOSS = nn.CrossEntropyLoss()
CORR_LOSS = Correlation(batch_size=batch_size).to(device)
kl_div_loss = nn.KLDivLoss(log_target=True)
accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

student = CI_model(input_size=im_size,
        snapshots=int(SPC_portion_st * 32 * 32),
        real="False").to(device)

teacher = CI_model(input_size=im_size,
        snapshots=int(SPC_portion_tchr * 32 * 32),
        real="False").to(device)

teacher.load_state_dict(torch.load(r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model.pth"))

for param in teacher.parameters():
    param.requires_grad = False

teacher.eval()
optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
wandb.init(project=f"KD_loss_deco_penalty_lambda1",name=f"Prueba {letter}{num_test} - penalty: {lambda1} _{SPC_portion_tchr*100}% CAP_T {SPC_portion_st*100}% CAP_S  _lr: {lr} _momentum: {momentum} _weight_decay: {weight_decay} _milestones: {milestones} _gamma: {gamma}",config={"num_epochs": num_epochs})

for epoch in range(num_epochs):
  student.train()

  train_loss = AverageMeter()
  train_acc = AverageMeter()

  train_labels_loss = AverageMeter()
  train_optics_loss = AverageMeter()
  train_kl_loss = AverageMeter()
  train_deco_loss = AverageMeter()

  data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="blue")
  for _, train_data in data_loop_train:
    x_imgs, x_labels = train_data
    x_imgs = x_imgs.to(device)
    x_labels = x_labels.to(device)

    x_hat_s, resnet_features_s = student(x_imgs)
    x_hat_t, resnet_features_t = teacher(x_imgs)

    pred_labels_s = torch.argmax(x_hat_s, dim=1)

    loss_deco = CE_LOSS(x_hat_s, x_labels)

    loss_optics = kd_rb_spc(
                pred_teacher=x_hat_t,
                pred_student=x_hat_s,
                loss_type="gram",
                ca_s=student.system_layer.H,
                ca_t=teacher.system_layer.H
                )

    loss_labels = CORR_LOSS(inputs=(x_hat_s, x_hat_t))

    temperature = 1

    soft_targets = nn.functional.log_softmax(x_hat_t / temperature, dim=-1)
    soft_prob = nn.functional.log_softmax(x_hat_s / temperature, dim=-1)
    loss_kl = kl_div_loss(soft_prob, soft_targets)

    loss_train = lambda1*loss_deco + lambda2*loss_optics + lambda3*loss_labels + lambda4*loss_kl     

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    train_loss.update(loss_train.item())
    train_deco_loss.update(loss_deco.item())
    train_optics_loss.update(loss_optics.item())
    train_kl_loss.update(loss_kl.item())
    train_labels_loss.update(loss_labels.item())
    train_acc.update(accuracy(pred_labels_s, x_labels).item())
    data_loop_train.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_train.set_postfix(loss=train_loss.avg, acc=train_acc.avg)
  
  with torch.no_grad():
    student.eval()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    val_labels_loss = AverageMeter()
    val_optics_loss = AverageMeter()
    val_kl_loss = AverageMeter()
    val_deco_loss = AverageMeter()

    data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour="green")
    for _, val_data in data_loop_val:
      x_imgs, x_labels = val_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)

      x_hat_s, resnet_features_s = student(x_imgs)
      x_hat_t, resnet_features_t = teacher(x_imgs)

      pred_labels_s = torch.argmax(x_hat_s, dim=1)
      loss_deco = CE_LOSS(x_hat_s, x_labels)

      loss_optics = kd_rb_spc(
                  pred_teacher=x_hat_t,
                  pred_student=x_hat_s,
                  loss_type="gram",
                  ca_s=student.system_layer.H,
                  ca_t=teacher.system_layer.H
                  )

      loss_labels = CORR_LOSS(inputs=(x_hat_s, x_hat_t))

      temperature = 1

      soft_targets = nn.functional.log_softmax(x_hat_t / temperature, dim=-1)
      soft_prob = nn.functional.log_softmax(x_hat_s / temperature, dim=-1)
      loss_kl = kl_div_loss(soft_prob, soft_targets)

      loss_val = lambda1*loss_deco + lambda2*loss_optics + lambda3*loss_labels + lambda4*loss_kl    

      val_loss.update(loss_val.item())
      val_deco_loss.update(loss_deco.item())
      val_optics_loss.update(loss_optics.item())
      val_kl_loss.update(loss_kl.item())
      val_labels_loss.update(loss_labels.item())
      val_acc.update(accuracy(pred_labels_s, x_labels).item())
      data_loop_val.set_description(f"Epoch: {epoch+1}/{num_epochs}")
      data_loop_val.set_postfix(loss=val_loss.avg, acc=val_acc.avg)
  
  scheduler.step()
  if val_acc.avg > current_acc:
            current_acc = val_acc.avg
            print(f"Saving model with Accuracy: {current_acc}")
            torch.save(student.state_dict(), f"{model_path}/model.pth")
            best_epoch = epoch

  wandb.log({"train_loss": train_loss.avg,
              "val_loss": val_loss.avg,
              "train_acc": train_acc.avg,
              "val_acc": val_acc.avg,
              "train_labels_loss": train_labels_loss.avg,
              "train_optics_loss": train_optics_loss.avg,
              "train_kl_loss": train_kl_loss.avg,
              "train_deco_loss": train_deco_loss.avg,
              "val_labels_loss": val_labels_loss.avg,
              "val_optics_loss": val_optics_loss.avg,
              "val_kl_loss": val_kl_loss.avg,
              "val_deco_loss": val_deco_loss.avg})

test_loss = AverageMeter()
test_acc = AverageMeter()

test_labels_loss = AverageMeter()
test_optics_loss = AverageMeter()
test_kl_loss = AverageMeter()
test_deco_loss = AverageMeter()

del student

student = CI_model(input_size=im_size,
        snapshots=int(SPC_portion_st * 32 * 32),
        real="False").to(device)

student.load_state_dict(torch.load(f"{model_path}/model.pth"))

data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="magenta")
with torch.no_grad():
  student.eval()
  for _, test_data in data_loop_test:
    x_imgs, x_labels = test_data
    x_imgs = x_imgs.to(device)
    x_labels = x_labels.to(device)

    x_hat_s, resnet_features_s = student(x_imgs)
    x_hat_t, resnet_features_t = teacher(x_imgs)

    pred_labels_s = torch.argmax(x_hat_s, dim=1)
    loss_deco = CE_LOSS(x_hat_s, x_labels)

    loss_optics = kd_rb_spc(
                pred_teacher=x_hat_t,
                pred_student=x_hat_s,
                loss_type="gram",
                ca_s=student.system_layer.H,
                ca_t=teacher.system_layer.H
                )

    loss_labels = CORR_LOSS(inputs=(x_hat_s, x_hat_t))

    temperature = 1

    soft_targets = nn.functional.log_softmax(x_hat_t / temperature, dim=-1)
    soft_prob = nn.functional.log_softmax(x_hat_s / temperature, dim=-1)
    loss_kl = kl_div_loss(soft_prob, soft_targets)

    loss_test = lambda1*loss_deco + lambda2*loss_optics + lambda3*loss_labels + lambda4*loss_kl    

    test_loss.update(loss_test.item())
    test_deco_loss.update(loss_deco.item())
    test_optics_loss.update(loss_optics.item())
    test_kl_loss.update(loss_kl.item())
    test_labels_loss.update(loss_labels.item())
    test_acc.update(accuracy(pred_labels_s, x_labels).item())
    data_loop_test.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_test.set_postfix(loss=test_loss.avg, acc=test_acc.avg)

wandb.log({"test_loss": test_loss.avg,
            "test_acc": test_acc.avg,
            "best_epoch": best_epoch,
            "test_labels_loss": test_labels_loss.avg,
            "test_optics_loss": test_optics_loss.avg,
            "test_kl_loss": test_kl_loss.avg,
            "test_deco_loss": test_deco_loss.avg})

wandb.finish()
