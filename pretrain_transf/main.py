import sys, torch, random
from tqdm import tqdm
sys.path.append("../")
sys.path.append("../Point_Transformers_master")
from TransformerModel import TransformerModel
from util import getDataloader

device = "cuda:0"
dataloader_train, dataloader_test = getDataloader(device)
model = TransformerModel(2048, nclass = 16).to(device)
opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
lossfn = torch.nn.CrossEntropyLoss()
epochs = 50

best = 0
train_log, eval_log = [], []
for epoch in range(epochs) :
    def run(dataloader, mode = "train") :
        acc, tot, tloss = 0, 0, 0
        iter = tqdm(dataloader)
        for x, y in iter : 
            pred = model(x)
            loss = lossfn(pred, y.squeeze(-1))

            if mode == "train" :
                opt.zero_grad()
                loss.backward()
                opt.step()

            acc += (pred.argmax(-1) == y.squeeze(-1)).sum().item()
            tot += y.shape[0]
            tloss += loss.item()
            iter.set_description("%s {epoch: %2d, acc: %.3f, loss: %4.4f}"%(mode, epoch, acc/tot, tloss/tot))
        return (acc/tot, tloss/tot)
    
    model.train()
    train_log.append(run(dataloader_train, "train"))
    with torch.no_grad() :
        model.eval()
        eval_log.append(run(dataloader_test, "eval"))
        if eval_log[-1][0] > best :
            best = eval_log[-1][0]
            torch.save(model, "pointtransformer.pt")


with open("log.log", "w") as f:
    f.write(str(train_log))
    f.write("\n")
    f.write(str(eval_log))