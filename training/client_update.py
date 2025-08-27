import torch.optim as optim
from utils.loss_function import distillation_loss, cross_entropy_loss
from models.student_model import StudentModel
from models.teacher_model import TeacherModel

def client_update(global_model, local_data, lambda_, T, tau):
    """Perform local training for a single client using a fresh teacher and student."""
    teacher = TeacherModel(global_model)
    student = StudentModel(global_model)
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
