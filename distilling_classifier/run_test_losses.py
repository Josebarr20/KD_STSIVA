import os

l1 = 1
lr = 0.1
batch_size = 2**7
num_epochs = 100
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1
SPC_portion_tchr = 0.2
SPC_portion_st = 0.1
dataset = "CIFAR10"
seed = 42
loss_response = "gram"
temperature = 1
save_path = "WEIGHTS/SPC_KD_TEST/"
teacher_path = r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model_t_CAP_20.pth"

c = 0
if c <= 3:
    l2 = 0
    l3_list = [0.2, 0.4, 0.6, 0.8]
    for l3 in l3_list:
        os.system(
            f"python main_cifar10.py --lambda1 {l1} --lambda2 {l2} --lambda3 {l3} --lr {lr} --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion_tchr {SPC_portion_tchr} --SPC_portion_st {SPC_portion_st} --dataset {dataset} --seed {seed} --loss_response {loss_response} --temperature {temperature} --save_path {save_path} --teacher_path {teacher_path}"
            )
        c += 1
else:
    l2_list = [0.2, 0.4, 0.6, 0.8]
    l3 = 0
    for l2 in l2_list:
        os.system(
            f"python main_cifar10.py --lambda1 {l1} --lambda2 {l2} --lambda3 {l3} --lr {lr} --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion_tchr {SPC_portion_tchr} --SPC_portion_st {SPC_portion_st} --dataset {dataset} --seed {seed} --loss_response {loss_response} --temperature {temperature} --save_path {save_path} --teacher_path {teacher_path}"
            )