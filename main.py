import torch
from cnn_model import cnn


model = cnn()
model.load_state_dict(torch.load('model.pth'))
model.eval()  