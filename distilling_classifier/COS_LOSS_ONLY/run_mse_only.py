import os
# os.chdir("distilling_classifier/MSE_LOSS_ONLY")

lambda2 = [0.25, 0.5, 0.75, 1] 
lambda1 = 1
students = [0.01]
teachers = [0.05, 0.1, 0.2]
epochs = 50
for st in students:
    for t in teachers:
        for l2 in lambda2:
            path = r"save_model_t\model_binary_"+str(int(t*100))+".pth"
            # print(f"Running experiment with SPC_portion_st: {st}, SPC_portion_tchr: {t}, lambda2: {l2}")
            os.system(f"python main_cifar10_KD_OP.py --num_epochs {epochs} --lambda1 1 --lambda2 {l2} --milestones 20 40 --SPC_portion_tchr {t} --SPC_portion_st {st} --teacher_path {path}")
