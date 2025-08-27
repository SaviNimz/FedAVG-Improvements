import torch
import torch.optim as optim
from utils.loss_function import distillation_loss, cross_entropy_loss

def client_update(student, teacher, local_data, lambda_, T, tau):
    """Perform local training for a single client."""
    teacher.freeze()
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=0.01)
    
    for inputs, labels in local_data:
        # Teacher predictions (frozen model)
        teacher_out = teacher(inputs)
        teacher_out = teacher_out / T  # Apply temperature scaling
        
        # Student predictions
        student_out = student(inputs)
        student_out = student_out / T
        
        # Calculate cross-entropy and knowledge distillation loss
        ce_loss = cross_entropy_loss(student_out, labels)
        kd_loss = distillation_loss(teacher_out, student_out, T)
        
        if max(teacher_out) >= tau:
            loss = ce_loss + lambda_ * kd_loss
        else:
            loss = ce_loss
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student.state_dict()
