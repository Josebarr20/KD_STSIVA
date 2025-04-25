import torch
import torch.nn as nn

def kd_rb_spc(loss_type, pred_teacher, pred_student, ca_s, ca_t):
    MSE = nn.MSELoss()

    if loss_type == "gram":
        #snapshots_s, M, N = ca_s.shape
        #snapshots_t, M, N = ca_t.shape

        #ca_s = ca_s.view(-1, M * N)
        #ca_t = ca_t.view(-1, M * N)

        gram_s = torch.matmul(ca_s.T, ca_s)

        gram_t = torch.matmul(ca_t.T, ca_t)

        return MSE(gram_s, gram_t)

    if loss_type not in ["gram"]:
        raise ValueError("Invalid loss type reconstruction")
    
class Correlation(nn.Module):
    r"""
    Correlation Regularization for the outputs of optical layers.

    This regularizer computes 

    .. math::
        \begin{equation*}
        R(\mathbf{y}_1,\mathbf{y}_2) = \mu\left\|\mathbf{C_{yy_1}} - \mathbf{C_{yy_2}}\right\|_2
        \end{equation*}
    
    where :math:`\mathbf{C_{yy_1}}` and :math:`\mathbf{C_{yy_2}}` are the correlation matrices of the measurements tensors :math:`\mathbf{y}_1,\mathbf{y}_2 \in \yset` and `\mu` is a regularization parameter .

    

    """

    def __init__(self, batch_size=128):
        """

        Args:
            batch_size (int): Batch size used for reshaping.
            param (float): Regularization parameter.
        """        
        super(Correlation, self).__init__()
        self.type_reg = 'measurements'

    def forward(self, inputs):
        """
        Compute correlation regularization term.

        Args:
            inputs (tuple): Tuple containing two input tensors (x and y).

        Returns:
            torch.Tensor: Correlation regularization term.
        """
        x, y = inputs
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, -1)
        y_reshaped = y.view(batch_size, -1)

        Cxx = torch.mm(x_reshaped, x_reshaped.t()) / batch_size
        Cyy = torch.mm(y_reshaped, y_reshaped.t()) / batch_size

        loss = torch.norm(Cxx - Cyy, p=2)
        return loss