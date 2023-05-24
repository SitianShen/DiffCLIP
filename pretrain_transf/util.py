import torch, numpy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
class PointDataset(Dataset):
    def __init__(self, x, y, device):
        super(PointDataset, self).__init__()
        self.x = x
        self.y = y
        self.len = len(x)
        self.device = device
    
    def __getitem__(self, i):
        x = torch.tensor(self.x[i].astype('float32'), device = self.device)
        y = torch.tensor(self.y[i].astype('int'), device = self.device)
        return x, y

    def __len__(self):
        return self.len
    
def getDataloader(device) :
    # return [(torch.randn(8, 2048, 3).to(device),torch.ones(8, dtype = int).to(device)),
    #           (torch.randn(8, 2048, 3).to(device),torch.zeros(8, dtype = int).to(device)),
    #           (torch.randn(8, 2048, 3).to(device),torch.ones(8, dtype = int).to(device)),
    #           (torch.randn(8, 2048, 3).to(device),torch.zeros(8, dtype = int).to(device)),
    #           (torch.randn(8, 2048, 3).to(device),torch.ones(8, dtype = int).to(device))
    #           ], None
    import h5py, os
    path = "../data/ShapeNet"
    train_list = ['ply_data_train0.h5', 'ply_data_train1.h5', 'ply_data_train2.h5', 'ply_data_train3.h5', 'ply_data_train4.h5', 'ply_data_train5.h5']
    val_list = ['ply_data_val0.h5']

    def load(file_list) :
        x = []
        y = []
        for file in tqdm(file_list):
            f = h5py.File(os.path.join(path, file), 'r')
            x.extend(f['data'])
            y.extend(f['label'])
            f.close()
        return PointDataset(x, y, device)       

    train = DataLoader(load(train_list), batch_size = 8, shuffle = True)
    val = DataLoader(load(val_list), batch_size = 8)

    return train, val
