import os

lr = 0.1
batch_size = 2**7
num_epochs = 1
momentum = 0.9
weight_decay = 5e-4  
gamma = 0.1
dataset = "CIFAR10"
seed = 42
loss_response = "gram"
temperature = 1
save_path = "WEIGHTS/SPC_KD_TEST/"
    
os.system(f"python main_cifar10.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.1} --dataset {dataset} --seed {seed}  --save_path {save_path}")
os.system(f"python main_cifar10.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.2} --dataset {dataset} --seed {seed}  --save_path {save_path}")
