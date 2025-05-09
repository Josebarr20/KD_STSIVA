import os

p1 = r"save_model_t\model_binary_100.pth"
lambda2 = [] 
for l2 in lambda2:
    os.system(f"python run_kl_only.py --lambda1 1 --lambda2 {l2} --milestones 20 40 --SPC_portion_tchr 1 --SPC_portion_st 0.1 --teacher_path {p1}")
