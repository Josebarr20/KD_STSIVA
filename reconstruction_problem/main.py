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
current_psnr = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = device = "cuda" if torch.cuda.is_available() else "cpu"

SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)
MSE_LOSS = nn.MSELoss()

num_epochs = 200
batch_size = 2**6
_, im_size, _, _, _, _, testloader, trainloader, valoader = get_dataset("MNIST", "data", batch_size, 42)

im_size = (1, im_size[-2], im_size[-1])

model = CI_model(input_size=im_size,
        snapshots=int(0.12 * 32 * 32),
        n_stages=7,
        device=device,
        SystemLayer="SPC",
        real="False",
        sampling_pattern=None,
        base_channels=64,
        decoder="unet",
        acc_factor=None).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)

wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
wandb.init(project="KD_MNIST", name="prueba 7",config={"num_epochs": num_epochs})

for epoch in range(num_epochs):
  model.train()

  train_loss = AverageMeter()
  train_ssim = AverageMeter()
  train_psnr = AverageMeter()

  data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="blue")
  for _, train_data in data_loop_train:
    x, _ = train_data
    x = x.to(device)
    x_hat, _, _, _ = model(x)
    loss_train = MSE_LOSS(x_hat, x)
    ssim_train = SSIM(x_hat, x)
    psnr_train = PSNR(x_hat, x)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    train_loss.update(loss_train.item())
    train_ssim.update(ssim_train.item())
    train_psnr.update(psnr_train.item())
    data_loop_train.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_train.set_postfix(loss=train_loss.avg, ssim=train_ssim.avg, psnr=train_psnr.avg)
  
  with torch.no_grad():
    model.eval()

    val_loss = AverageMeter()
    val_ssim = AverageMeter()
    val_psnr = AverageMeter()

    data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour="magenta")
    for _, val_data in data_loop_val:
      x, _ = val_data
      x = x.to(device)
      x_hat, _, _, _ = model(x)
      loss_val = MSE_LOSS(x_hat, x)
      ssim_val = SSIM(x_hat, x)
      psnr_val = PSNR(x_hat, x)
      val_loss.update(loss_val.item())
      val_ssim.update(ssim_val.item())
      val_psnr.update(psnr_val.item())
      data_loop_val.set_description(f"Epoch: {epoch+1}/{num_epochs}")
      data_loop_val.set_postfix(loss=val_loss.avg, ssim=val_ssim.avg, psnr=val_psnr.avg)

  if val_psnr.avg > current_psnr:
            current_psnr = val_psnr.avg
            print(f"Saving model with PSNR: {current_psnr}")
            torch.save(model.state_dict(), f"{model_path}/model.pth")

  iamge_array = save_coded_apertures(model.system_layer, 8, 2, images_path, f"coded_aperture_{epoch}", "SPC")
  images = wandb.Image(iamge_array, caption=f"Epoch: {epoch}")

  recs_array, psnr_imgs, ssim_imgs = save_reconstructed_images(
          x, x_hat, 3, 2, images_path, f"reconstructed_images_{epoch}", PSNR, SSIM
      )
  recs_images = wandb.Image(recs_array, caption=f"Epoch: {epoch}\nReal\nRec\nPSNRs: {psnr_imgs}\nSSIMs: {ssim_imgs}")

  wandb.log({"train_loss": train_loss.avg,
              "train_ssim": train_ssim.avg,
              "train_psnr": train_psnr.avg,
              "val_loss": val_loss.avg,
              "val_ssim": val_ssim.avg,
              "val_psnr": val_psnr.avg,
              "coded_aperture": images,
              "reconstructed": recs_images})

test_loss = AverageMeter()
test_ssim = AverageMeter()
test_psnr = AverageMeter()  

del model

model = CI_model(input_size=im_size,
        snapshots=int(0.12 * 32 * 32),
        n_stages=7,
        device=device,
        SystemLayer="SPC",
        real="hadamard",
        sampling_pattern=None,
        base_channels=64,
        decoder="unet",
        acc_factor=None).to(device)

model.load_state_dict(torch.load(f"{model_path}/model.pth"))

data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="green")
with torch.no_grad():
  model.eval()
  for _, test_data in data_loop_test:
    x, _ = test_data
    x = x.to(device)
    x_hat, _, _, _ = model(x)
    loss_test = MSE_LOSS(x_hat, x)
    ssim_test = SSIM(x_hat, x)
    psnr_test = PSNR(x_hat, x)
    test_loss.update(loss_test.item())
    test_ssim.update(ssim_test.item())
    test_psnr.update(psnr_test.item())
    data_loop_test.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_test.set_postfix(loss=test_loss.avg, ssim=test_ssim.avg, psnr=test_psnr.avg)

wandb.log({"test_loss": test_loss.avg,
          "test_ssim": test_ssim.avg,
          "test_psnr": test_psnr.avg})

wandb.finish()