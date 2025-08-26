import torch
import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(self, global_model):
        super(TeacherModel, self).__init__()
        # Freeze the global model to create the teacher model
        self.model = global_model
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze the teacher model
        
    def forward(self, x):
        return self.model(x)
