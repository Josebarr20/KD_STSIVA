import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from models import CI_model
from utils import (get_dataset, 
                   set_seed, 
                   save_metrics, 
                   AverageMeter, 
                   save_coded_apertures, 
                   save_reconstructed_images)

set_seed(42)

images_path, model_path = save_metrics(f"imagenes_kd")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = device = "cuda" if torch.cuda.is_available() else "cpu"

SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)
CE_LOSS = nn.CrossEntropyLoss()

num_epochs = 100
batch_size = 2**6
_, im_size, _, _, _, _, testloader, trainloader, valoader = get_dataset("MNIST", "data", batch_size, 42)

im_size = (1, im_size[-2], im_size[-1])

model = CI_model(input_size=im_size,
        snapshots=int(0.12 * 32 * 32),
        n_stages=None,
        device=device,
        SystemLayer="SPC",
        real="hadamard",
        sampling_pattern=None,
        base_channels=None,
        decoder="resnet18",
        acc_factor=None).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
wandb.init(project="Classification_MNIST", name="prueba 2",config={"num_epochs": num_epochs})

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
    loss_train = CE_LOSS(x_hat[0], x_labels)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    train_loss.update(loss_train.item())
    data_loop_train.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_train.set_postfix(loss=train_loss.avg,)
  
  with torch.no_grad():
    model.eval()

    val_loss = AverageMeter()

    data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour="magenta")
    for _, val_data in data_loop_val:
      x_imgs, x_labels = val_data
      x_imgs = x_imgs.to(device)
      x_labels = x_labels.to(device)  
      x_hat = model(x_imgs)
      loss_val = CE_LOSS(x_hat[0], x_labels)
      val_loss.update(loss_val.item())
      data_loop_train.set_description(f"Epoch: {epoch+1}/{num_epochs}")
      data_loop_train.set_postfix(loss=val_loss.avg)

  iamge_array = save_coded_apertures(model.system_layer, 8, 2, images_path, f"coded_aperture_{epoch}", "SPC")
  images = wandb.Image(iamge_array, caption=f"Epoch: {epoch}")

  wandb.log({"train_loss": train_loss.avg,
              "val_loss": val_loss.avg,
              "coded_aperture": images})

test_loss = AverageMeter()

del model

model = CI_model(input_size=im_size,
        snapshots=int(0.12 * 32 * 32),
        n_stages=None,
        device=device,
        SystemLayer="SPC",
        real="hadamard",
        sampling_pattern=None,
        base_channels=None,
        decoder="resnet18",
        acc_factor=None).to(device)

model.load_state_dict(torch.load(f"{model_path}/model.pth"), strict=False)

data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="green")
with torch.no_grad():
  model.eval()
  for _, test_data in data_loop_test:
    x_imgs, x_labels = val_data
    x_imgs = x_imgs.to(device)
    x_labels = x_labels.to(device)  
    x_hat = model(x_imgs)
    loss_test = CE_LOSS(x_hat[0], x_labels)
    test_loss.update(loss_test.item())

    data_loop_test.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_test.set_postfix(loss=test_loss.avg)

wandb.log({"test_loss": test_loss.avg})

wandb.finish()