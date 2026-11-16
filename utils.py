import torch
import torch.nn as nn
import torch.optim as optim
class Red(nn.Module):
    def __init__(self,input_dim:int=128,output_dim:int=5):
        super().__init__()
        capa1=nn.Linear(input_dim,256)
        capa2=nn.Linear(256,128)
        capa3=nn.Linear(128,32)
        capa4=nn.Linear(32,output_dim)

        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        x = self.relu(self.capa1(x))
        x = self.relu(self.capa2(x))
        x = self.relu(self.capa3(x))
        return self.capa4(x)

    