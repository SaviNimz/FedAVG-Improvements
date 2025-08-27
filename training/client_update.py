import torch
import torch.optim as optim
from utils.loss_function import distillation_loss, cross_entropy_loss

def client_update(student, teacher, local_data, lambda_, T, tau):
    """Perform local training for a single client."""
    teacher.freeze()
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=0.01)
    
    for inputs, labels in local_data:
        # Teacher and student predictions (raw logits)
        teacher_out = teacher(inputs)
        student_out = student(inputs)

        # Calculate cross-entropy on raw student logits
        ce_loss = cross_entropy_loss(student_out, labels)
        kd_loss = distillation_loss(teacher_out, student_out, T)

        # Apply distillation only when teacher is confident
        if torch.max(torch.softmax(teacher_out / T, dim=1)) >= tau:
            loss = ce_loss + lambda_ * kd_loss
        else:
            loss = ce_loss
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student.state_dict()
