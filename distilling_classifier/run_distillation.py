import os

# p1 = r"C:\Users\USUARIO\Documents\STISIV2025\KD_STSIVA\distilling_classifier\save_model_t\model_binary_100.pth"
p1 = r"save_model_t\model_binary_100.pth"
for lambda3 in [0.2, 0.4, 0.6, 0.8]:
    os.system(f"python main_cifar10_KD.py --lambda1 1 --lambda3 {lambda3} --milestones 30 50 70 80 --SPC_portion_tchr 1 --SPC_portion_st 0.1 --teacher_path {p1} --real_st False --real_tchr False --project_name KD_Temp_study")
