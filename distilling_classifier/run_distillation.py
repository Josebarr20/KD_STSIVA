import os

l1 = 1
p1 = r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model_binary_100.pth"
p2 = r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model_binary_80.pth"
p3 = r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model_binary_60.pth"
p4 = r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model_binary_40.pth"
p5 = r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model_binary_20.pth"
p6 = r"C:\Users\SERGIOURREA\Desktop\KD_Jose\distilling_classifier\save_model_t\model_binary_10.pth"

# Optics Loss

os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.2} --milestones 30 50 70 80 --SPC_portion_tchr {0.4} --SPC_portion_st {0.1} --teacher_path {p4} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.4} --milestones 30 50 70 80 --SPC_portion_tchr {0.4} --SPC_portion_st {0.1} --teacher_path {p4} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.6} --milestones 30 50 70 80 --SPC_portion_tchr {0.4} --SPC_portion_st {0.1} --teacher_path {p4} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.8} --milestones 30 50 70 80 --SPC_portion_tchr {0.4} --SPC_portion_st {0.1} --teacher_path {p4} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")

os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.2} --milestones 30 50 70 80 --SPC_portion_tchr {0.2} --SPC_portion_st {0.1} --teacher_path {p5} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.4} --milestones 30 50 70 80 --SPC_portion_tchr {0.2} --SPC_portion_st {0.1} --teacher_path {p5} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.6} --milestones 30 50 70 80 --SPC_portion_tchr {0.2} --SPC_portion_st {0.1} --teacher_path {p5} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.8} --milestones 30 50 70 80 --SPC_portion_tchr {0.2} --SPC_portion_st {0.1} --teacher_path {p5} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")

os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.2} --milestones 30 50 70 80 --SPC_portion_tchr {0.1} --SPC_portion_st {0.1} --teacher_path {p6} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.4} --milestones 30 50 70 80 --SPC_portion_tchr {0.1} --SPC_portion_st {0.1} --teacher_path {p6} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.6} --milestones 30 50 70 80 --SPC_portion_tchr {0.1} --SPC_portion_st {0.1} --teacher_path {p6} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
os.system(f"python main_cifar10_KD.py --lambda1 {l1} --lambda3 {0} --lambda2 {0.8} --milestones 30 50 70 80 --SPC_portion_tchr {0.1} --SPC_portion_st {0.1} --teacher_path {p6} --real_st {"False"} --real_tchr {"False"} --project_name {"KD_Losses_B-B"}")
