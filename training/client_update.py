import torch.optim as optim
import torch
from utils.loss_function import distillation_loss, cross_entropy_loss
from models.student_model import StudentModel
from models.teacher_model import TeacherModel

def client_update(global_model, local_data, lambda_, T, tau, learning_rate):
    """Perform local training for a single client using a fresh teacher and student.

    Parameters:
        global_model: The global model to be used as a starting point for the student and teacher models.
        local_data: The data loader containing the client's local data.
        lambda_ (float): Weight for the distillation loss component.
        T (float): Temperature parameter for knowledge distillation.
        tau (float): Confidence threshold for applying distillation.
        learning_rate (float): Learning rate for the student's optimizer.
    """
    teacher = TeacherModel(global_model)
    student = StudentModel(global_model)
    teacher.freeze()
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=learning_rate)

    for inputs, labels in local_data:
        # Teacher and student predictions (raw logits)
        teacher_out = teacher(inputs)
        student_out = student(inputs)

        # Calculate cross-entropy on raw student logits
        ce_loss = cross_entropy_loss(student_out, labels)

        # Compute teacher probabilities and confidence
        teacher_probs = torch.softmax(teacher_out / T, dim=1)
        conf, _ = teacher_probs.max(dim=1)

        kd_loss = distillation_loss(teacher_out, student_out, T)

        # Apply distillation only when average confidence exceeds the threshold
        if conf.mean() >= tau:
            loss = ce_loss + lambda_ * kd_loss
        else:
            loss = ce_loss
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student.state_dict()
