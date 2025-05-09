import os
# os.chdir("distilling_classifier/MSE_LOSS_ONLY")

lambda3 = [0.25, 0.5, 0.75, 1] 
epochs = 50
temperatura = [2, 4, 6]

for l3 in lambda3:
    for temp in temperatura:
        path = r"save_model_t/model_binary_10.pth"
        # print(f"Running experiment with SPC_portion_st: {0.05}, SPC_portion_tchr: {0.01}, lambda2: {l2}, temp: {temp}")
        os.system(f"python main_cifar10_KD_N.py --num_epochs {epochs} --lambda1 1 --lambda3 {l3} --milestones 20 40 --SPC_portion_tchr {0.1} --SPC_portion_st {0.01} --teacher_path {path}")
