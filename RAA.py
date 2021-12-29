import numpy as np
import torch 
import torch.nn as nn
criterion=nn.CrossEntropyLoss()
device = torch.device('cuda')
# Function to implement RAA attack 
#Input: Input(x), Class Label(y), epsilon, Gradient: x_grad 
#Output: Adversarial Sample(Adv_x)
def RAA_Attack(x,epsilon,x_grad):
  Adv_x = x - epsilon*x_grad.sign()
  Adv_x = torch.clamp(Adv_x, 0, 1)
  return Adv_x
for (x, y) in Data_Loader:
      # Send data to device
      x,y = x.to(device), y.to(device)
      x.requires_grad = True
      # Predict class label of input x
      Predicted_Label = model(x)
      Initial_prediction = Predicted_Label.max(1, keepdim=True)[1] 
      # Compute Cross-entropy loss
      loss = criterion(Initial_Prediction,y)
      model.zero_grad()
      loss.backward()
      # Compute Gradient w.r.t. loss
      x_grad = x.grad.data
      # Generate adversarial sample
      Adv_x=RAA_Attack(x,epsilon,x_grad)
