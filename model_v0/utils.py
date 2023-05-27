from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import numpy, sys, random


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class default_MainDataset(Dataset) :
    def __init__(self, x, y, projectionModel, path = "../data/ModelNet10/", train_skip_rate = 0) :
        super(default_MainDataset, self).__init__()
        self.path = path
        self.filename = x
        self.y = y
        self.model = projectionModel
        self.len = len(y)
        self.train_skip_rate = train_skip_rate

    def __getitem__(self, id):
        print("ingetit", self.path + self.filename[id])
        if random.random() < self.train_skip_rate : 
            print("outgetit")
            return None, None, None, None
        with open(self.path + self.filename[id]) as f:
            line = f.readline().strip()
            if len(line.split(" ")) > 1 :
                line = line[3:].split(" ")
            else :
                line = f.readline().strip().split(" ")
            n, m, _ = map(int, line)
            pointList, planeList = [], []
            for i in range(n) :
                x, y, z = map(float, f.readline().strip().split(" "))
                pointList.append([x, y, z])
            for i in range(m) :
                _, a, b, c = map(int, f.readline().strip().split(" "))
                planeList.append([a, b, c])
            
        pic, multi, sample_points = self.model(numpy.array(pointList), planeList)
        # from PIL import Image
        # img = Image.fromarray(pic[0, ...].squeeze())
        # img.save("./autodensitytry/%d'0-project-example.png"%id)
        # img = Image.fromarray(pic[1, ...].squeeze())
        # img.save("./autodensitytry/%d'1-project-example.png"%id)
        # img = Image.fromarray(pic[2, ...].squeeze())
        # img.save("./autodensitytry/%d'2-project-example.png"%id)
        # img = Image.fromarray(pic[3, ...].squeeze())
        # img.save("./autodensitytry/%d'3-project-example.png"%id)
        # print("outgetit")
        return pic, multi, sample_points.astype('float32')[None, ...], self.y[id]


    def __len__(self):
        return self.len

def default_collate_fn(x) :
    pic, multi, sample_points, y = x[0]
    return pic, multi, sample_points, y

def getDataloader(projectionModel, ntrain = 8, ntest = 3, path = "../data/ModelNet10/", collate_fn=default_collate_fn, MainDataset = default_MainDataset) :
    import os

    classes = []
    for file in os.listdir(path) :
        if os.path.isdir(path + file) :
            classes.append(file)
    print(classes)
    def get(mode, nkeep, train_skip_rate = 0) :
        x, y = [], []
        nkeep_mem = nkeep
        for C in range(len(classes)) :
            files = os.listdir(path + classes[C] + "/%s/"%mode)
            if nkeep_mem == -1 : nkeep = len(files)
            skiprate, counter = len(files)//nkeep, 0
            print(len(files), skiprate, nkeep)
            for file in files :
                if file[-1] != 'f' : continue
                counter += 1
                if (counter % skiprate != 0) :continue
                x.append(classes[C] + "/%s/"%mode + file)
                y.append(C)
        random = False if mode == "test" else True
        return DataLoaderX(
                    MainDataset(
                            x, y, 
                            projectionModel, 
                            path,
                            train_skip_rate
                    ),
                    collate_fn=collate_fn,
                    shuffle=random,
                    num_workers = 2
        )
    return get("train", ntrain, 0.0), get("test", ntest), classes

def classificationFn(mylogits_per_image, multi) :
    import torch
    mylogits_per_image = torch.nn.functional.max_pool2d(
        mylogits_per_image[None, ...], (multi, 1)
    )[0, ...]

    probs = mylogits_per_image.softmax(dim=-1)

    log_probs = torch.log(probs)
    topers = (log_probs.sort(dim=0)[0])[:, :]
    global_info = topers.mean(dim=0)
    expglobal_info = torch.exp(global_info)
    # global_info = global_info.softmax(dim=-1)
    expglobal_info1 = expglobal_info - expglobal_info.min().detach()
    expglobal_info2 = expglobal_info1 / expglobal_info1.max().detach()

    # print(expglobal_info2)
    # print(probs.max(dim=0)[0])

    pred = expglobal_info2*probs.max(dim=0)[0]

    return pred


# from Models import ProjectionModel
# if __name__ == '__main__':
#     d = MainDataset(["bed/train/bed_0001.off", "bed/train/bed_0002.off"], [0, 1], ProjectionModel.Perspective4view(512)) 
#     l = DataLoader(d, collate_fn=collate_fn)

#     for pic, multi, y in l :
#         print(pic.shape, multi, y, sep="\n")
