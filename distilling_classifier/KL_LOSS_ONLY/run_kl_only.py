import os

p1 = r"save_model_t\model_binary_100.pth"
lambda3 = [] 
for l3 in lambda3:
    os.system(f"python run_kl_only.py --lambda1 1 --lambda3 {l3} --milestones 20 40 --SPC_portion_tchr 1 --SPC_portion_st 0.1 --teacher_path {p1}")
