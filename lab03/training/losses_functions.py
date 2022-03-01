import torch
import torch.nn as nn
from typing import Type

class StudentLossFunction(nn.Module):
    """Implementation of the component of loss function which uses soft labels (Kullback Leibler Divergence of the
        softmax predictions of the student and teacher networks)        
    """
    def __init__(self):
        super().__init__()

        self.kld_loss = nn.KLDivLoss(log_target=False, reduction='batchmean')

    def forward(self, student_output: Type[torch.Tensor], teacher_output: Type[torch.Tensor]) -> torch.Tensor:
        return self.kld_loss(student_output, teacher_output).sum(dim=0)

def get_teacher_loss_function():
    return nn.CrossEntropyLoss()

def get_student_loss_function():
    return StudentLossFunction()
