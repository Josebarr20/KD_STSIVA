import torch
import torch.nn as nn

def kd_rb_spc(y_teacher, y_student, pred_teacher, pred_student, loss_type, ca_s, ca_t):
    MSE = nn.MSELoss()

    if loss_type == "gram":
        snapshots_s, M, N = ca_s.shape
        snapshots_t, M, N = ca_t.shape

        ca_s = ca_s.view(-1, M * N)
        ca_t = ca_t.view(-1, M * N)

        gram_s = torch.matmul(ca_s.T, ca_s)

        gram_t = torch.matmul(ca_t.T, ca_t)

        return MSE(gram_s, gram_t)

    if loss_type not in ["gram"]:
        raise ValueError("Invalid loss type reconstruction")


def kd_ft_spc(
    loss_type: str,
    feats_teacher,
    feats_student,
    decoder,
):
    MSE = nn.MSELoss()

    if loss_type == "3":
        bottleneck_teacher = feats_teacher[3]
        bottleneck_student = feats_student[3]

        return MSE(bottleneck_student, bottleneck_teacher)

    if loss_type == "nothing":
        return torch.tensor(0)

    if loss_type not in [
        "3",
        "nothing",
    ]:
        raise ValueError("Invalid loss type features")
