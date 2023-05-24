import torch
from Point_Transformers_master.models.model import TransitionDown
from Point_Transformers_master.models.transformer import TransformerBlock


class Backbone(torch.nn.Module):
    def __init__(self, npoints, nblocks = 4, nneighbor = 16, d_points = 3, transformer_dim = 512):
        super(Backbone, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(d_points, 32),
            torch.nn.ReLU(),
           torch. nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = torch.nn.ModuleList()
        self.transformers = torch.nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats

class TransformerModel(torch.nn.Module) :
    def __init__(self, npoints, nclass, nblocks = 4, nneighbor = 16, d_points = 3, transformer_dim = 512, fromPretrainPath = None):
        super(TransformerModel, self).__init__()
        self.backbone = Backbone(npoints, nblocks, nneighbor, d_points, transformer_dim)
        if (fromPretrainPath is not None) :
            self.backbone = torch.load(fromPretrainPath)
        
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(32 * 2 ** nblocks, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, nclass)
        )
    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        return res

        
