import torch
import torch.optim as optim
from models.student_model import StudentModel
from models.teacher_model import TeacherModel
from utils.loss_function import distillation_loss, cross_entropy_loss

def client_update(global_model, local_data, lambda_, T, tau):
    """
    Performs local training of the student model using knowledge distillation and cross-entropy loss.
    
    Parameters:
        global_model (nn.Module): The global model received from the server (teacher model).
        local_data (DataLoader): The local data for training the student model.
        lambda_ (float): Weight of the knowledge distillation loss.
        T (float): Temperature parameter for knowledge distillation.
        tau (float): Confidence threshold for skipping distillation loss on unreliable teacher predictions.
    
    Returns:
        torch.Tensor: Updated model weights for the student.
    """
    # Clone the global model into a frozen teacher and a trainable student
    teacher = TeacherModel(global_model).freeze()
    student = StudentModel(global_model).train()
    
    # Optimizer for the student model
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
