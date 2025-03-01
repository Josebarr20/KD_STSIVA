import e2e
import load_data
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

batch_size = 128
channel,im_size,num_classes,class_names,x_train,x_test,testloader,trainloader,valoader,=load_data.get_dataset('MNIST',"data",batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = e2e.E2E(pinv=False, num_measurements=122, img_size=(32, 32), trainable=True, num_channels=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

num_epochs = 500
wandb.init(name="Prueba Final VScode",project="SPC_E2E_MNIST", config={
    "num_epochs": num_epochs})

ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
# Train stage
model.train()
for epoch in range(num_epochs):
  data_loop_train = tqdm(enumerate(testloader), total=len(testloader), colour="blue")
  for i, (y,_) in data_loop_train:
    y = y.to(device)
    x_hat = model(y).to(device)
    train_loss = criterion(x_hat, y)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    train_ssim = ssim(x_hat, y).to(device)
    train_psnr = psnr(x_hat, y).to(device)
    data_loop_train.set_description(f"Epoch: {epoch+1}/{num_epochs}")
    data_loop_train.set_postfix(loss=train_loss, ssim=train_ssim, psnr=train_psnr)

  # Validation stage
  model.eval()
  with torch.no_grad():
    data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour="green")
    for i, (y,_) in data_loop_val:
      y = y.to(device)
      x_hat = model(y).to(device)
      val_loss = criterion(x_hat, y).to(device)
      val_ssim = ssim(x_hat, y).to(device)
      val_psnr = psnr(x_hat, y).to(device)
      data_loop_val.set_description(f"Epoch: {epoch+1}/{num_epochs}")
      data_loop_val.set_postfix(loss=val_loss, ssim=val_ssim, psnr=val_psnr)

  if (epoch + 1) % 10 == 0:
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(y[0].squeeze().squeeze().cpu().detach().numpy())
    aux = x_hat[0].squeeze().squeeze().cpu().detach().numpy()
    axis[1].imshow(aux)
    plt.show()
    wandb_img = wandb.Image(fig)

  wandb.log({"epoch": epoch,
             "train_loss": train_loss.item(),
             "val_loss": val_loss.item(),
             "train_ssim": train_ssim.item(),
             "val_ssim": val_ssim.item(),
             "train_psnr": train_psnr.item(),
             "val_psnr": val_psnr.item(),
             "image": wandb_img if (epoch + 1) % 10 == 0 else None})

wandb.finish()