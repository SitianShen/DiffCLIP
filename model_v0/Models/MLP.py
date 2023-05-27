import torch
class MLP(torch.nn.Module) :
    def __init__(self, N, dim, device = "cuda:0") :
        super(MLP, self).__init__()
        self.dim = dim
        self.dropout = torch.nn.Dropout(0.5)
        self.f = torch.nn.ReLU()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dim, dim, device=device)
            for _ in range(N)
        ])
        self.lst = torch.nn.Linear(dim, dim, bias = False, device=device)
        self.layer = torch.nn.LayerNorm(dim)

    def forward(self, x) :
        for layer in self.layers : 
            x = self.f(self.dropout(layer(x)))
        out = self.f(self.lst(x))
        return self.layer(out)
        