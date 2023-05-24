import torch
class CosineSim(torch.nn.Module) :
    def __init__(self) :
        super(CosineSim, self).__init__()
        
    def forward(self, x: torch.tensor, y: torch.tensor) :
        return x*y/x.norm(p=2)/y.norm(p=2)