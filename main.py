import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from e2e import E2E
from utils import (get_dataset, 
                   set_seed, 
                   save_metrics, 
                   AverageMeter, 
                   save_coded_apertures, 
                   save_reconstructed_images)
from models import CI_model

def main(args):
  set_seed(42)

  images_path, _, _ = save_metrics(f"imagenes_kd")

  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  device = device = "cuda" if torch.cuda.is_available() else "cpu"

  SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
  PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)
  MSE_LOSS = nn.MSELoss()

  num_epochs = 250
  batch_size = 64
  _, im_size, _, _, _, _, testloader, trainloader, valoader = get_dataset(
    "MNIST", "data", batch_size, 42)
  im_size = (1, im_size[-2], im_size[-1])
  

  model = E2E(pinv=False, num_measurements=122, img_size=(32, 32), trainable=True, num_channels=1).to(device)
  MSE_LOSS = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=5e-4)
  
  wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
  wandb.init(project="prueba 1", name="KD_MNIST", config=args)

  for epoch in range(num_epochs):
    model.train()

    train_loss = AverageMeter()
    train_ssim = AverageMeter()
    train_psnr = AverageMeter()

    data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="blue")
    for _, train_data in data_loop_train:
      x, _ = train_data
      x = x.to(device)
      x_hat = model(x).to(device)
      loss_train = MSE_LOSS(x_hat, x)

      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()

      train_loss.update(loss_train.item())
      train_ssim.update(SSIM(x_hat, x).item())
      train_psnr.update(PSNR(x_hat, x).item())
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
        x_hat = model(x).to(device)
        loss_val = MSE_LOSS(x_hat, x)
        val_loss.update(loss_val.item())
        val_ssim.update(SSIM(x_hat, x).item())
        val_psnr.update(PSNR(x_hat, x).item())
        data_loop_train.set_description(f"Epoch: {epoch+1}/{num_epochs}")
        data_loop_train.set_postfix(loss=val_loss.avg, ssim=val_ssim.avg, psnr=val_psnr.avg)

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
  
  model = E2E(pinv=False, num_measurements=122, img_size=(32, 32), trainable=False, num_channels=1).to(device)

  with torch.no_grad():
    model.eval()
    data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="green")
    for _, test_data in data_loop_test:
      x, _ = test_data
      x = x.to(device)
      x_hat = model(x).to(device)
      loss_test = MSE_LOSS(x_hat, x)
      test_loss.update(loss_test.item())
      test_ssim.update(SSIM(x_hat, x).item())
      test_psnr.update(PSNR(x_hat, x).item())
      data_loop_test.set_description(f"Epoch: {epoch+1}/{num_epochs}")
      data_loop_test.set_postfix(loss=test_loss.avg, ssim=test_ssim.avg, psnr=test_psnr.avg)
  
  wandb.log({"test_loss": test_loss.avg,
            "test_ssim": test_ssim.avg,
            "test_psnr": test_psnr.avg})
  
  wandb.finish()