import os

# lambda2 = [0.25, 0.5, 0.75, 1] 
lambda1 = 1
# students = [0.01, 0.05]
# teachers = [0.1, 0.2, 0.4]
epochs = 1

p4 = r"save_model_t\model_binary_40.pth"
p1 = r"save_model_t\model_binary_10.pth"
p2 = r"save_model_t\model_binary_20.pth"

os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.25} --milestones 20 40 --SPC_portion_tchr {0.4} --SPC_portion_st {0.01} --teacher_path {p4}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.5} --milestones 20 40 --SPC_portion_tchr {0.4} --SPC_portion_st {0.01} --teacher_path {p4}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.75} --milestones 20 40 --SPC_portion_tchr {0.4} --SPC_portion_st {0.01} --teacher_path {p4}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {1.0} --milestones 20 40 --SPC_portion_tchr {0.4} --SPC_portion_st {0.01} --teacher_path {p4}")

os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.25} --milestones 20 40 --SPC_portion_tchr {0.1} --SPC_portion_st {0.05} --teacher_path {p1}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.5} --milestones 20 40 --SPC_portion_tchr {0.1} --SPC_portion_st {0.05} --teacher_path {p1}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.75} --milestones 20 40 --SPC_portion_tchr {0.1} --SPC_portion_st {0.05} --teacher_path {p1}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {1.0} --milestones 20 40 --SPC_portion_tchr {0.1} --SPC_portion_st {0.05} --teacher_path {p1}")

os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.25} --milestones 20 40 --SPC_portion_tchr {0.2} --SPC_portion_st {0.05} --teacher_path {p2}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.5} --milestones 20 40 --SPC_portion_tchr {0.2} --SPC_portion_st {0.05} --teacher_path {p2}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {0.75} --milestones 20 40 --SPC_portion_tchr {0.2} --SPC_portion_st {0.05} --teacher_path {p2}")
os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 {lambda1} --lambda2 {1.0} --milestones 20 40 --SPC_portion_tchr {0.2} --SPC_portion_st {0.05} --teacher_path {p2}")


