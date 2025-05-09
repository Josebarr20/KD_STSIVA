import os

p1 = r"save_model_t\model_binary_5.pth"
lambda2 = [0.25, 0.5, 0.75, 1] 
for l2 in lambda2:
    os.system(f"python run_kl_only.py --lambda1 1 --lambda2 {l2} --milestones 20 40 --SPC_portion_tchr 0.5 --SPC_portion_st 0.01 --teacher_path {p1}")
