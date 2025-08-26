import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, global_model):
        super(StudentModel, self).__init__()
        # Inherit the architecture from the global model
        self.model = global_model
        
    def forward(self, x):
        return self.model(x)
