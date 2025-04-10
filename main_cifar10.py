import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from models import CI_model
from utils import (get_dataset, 
                   set_seed, 
                   save_metrics, 
                   AverageMeter)

set_seed(42)

images_path, model_path = save_metrics(f"imagenes_kd")
current_acc = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 50
batch_size = 32

_, im_size, num_classes, class_names, _, _, testloader, trainloader, valoader = get_dataset("CIFAR10", "data", batch_size, 42)

im_size = (3, im_size[-2], im_size[-1])

model = CI_model(input_size=im_size,
        snapshots=int(1 * 32 * 32),
        real="False").to(device)

CE_LOSS = nn.CrossEntropyLoss()
accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5) 

wandb.login(key="279b022981698aa4f3b2c4cd1454d16ea345b195")

wandb.init(
    project="Classification_CIFAR10_Teacher",
    name="Prueba dumy - 100% CAP_test lr = {} - batch_size = {}".format(1e-5, batch_size),
    config={"num_epochs": num_epochs}
)



import torchvision.transforms.functional as TF

sample_batch = next(iter(trainloader))
images, labels = sample_batch


plt.figure(figsize=(12, 4))
for i in range(6):
    img = images[i]  
    img = TF.to_pil_image(img) 
    plt.subplot(1, 6, i+1)
    plt.imshow(img)
    plt.title(f"Label: {class_names[labels[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

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
  
  if val_acc.avg > current_acc:
            current_acc = val_acc.avg
            print(f"Saving model with Accuracy: {current_acc}")
            torch.save(model.state_dict(), f"{model_path}/model.pth")

  wandb.log({"train_loss": train_loss.avg,
              "val_loss": val_loss.avg,
              "train_acc": train_acc.avg,
              "val_acc": val_acc.avg})

test_loss = AverageMeter()
test_acc = AverageMeter()

del model

model = CI_model(input_size=im_size,
        snapshots=int(1 * 32 * 32),
        real="False").to(device)



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
            "test_acc": test_acc.avg})

wandb.finish()