import os

# p1 = r"C:\Users\USUARIO\Documents\STISIV2025\KD_STSIVA\distilling_classifier\save_model_t\model_binary_100.pth"

os.chdir("distilling_classifier")
# Confirmar que el cambio fue exitoso
print("Directorio actual:", os.getcwd())
p1 = r"save_model_t\model_binary_100.pth"
lambda1s = [0.5, 0.8, 1] #task
lambda2s = [0.3, 0.6, 1] #similarity
lambda3 = 0
for lambda1 in lambda1s:
    for lambda2 in lambda2s:
        os.system(f"python main_cifar10_KD.py --lambda1 {lambda1} --lambda2 {lambda2} --milestones 20 40 --SPC_portion_tchr 1 --SPC_portion_st 0.1 --teacher_path {p1} --real_st False --real_tchr False --project_name PCA_Similarity_LOSS_ONLY")
