def train(loadPath = None):
    import random
    from tqdm import tqdm
    
    clipmodel, preprocess = clip.load("ViT-B-32", device=device,download_root = "../../encoder_model")
    projectionModel = ProjectionModel.Perspective4view_plus_autodensity(
        image_resolution = 512,
        rotate = (-0, -35, -135),
        npoints = 2048,
        ndensity = (6000000,300000)
    )    
    multi = 4
    dataloader_train, dataloader_test, classes = utils.getDataloader(
        projectionModel,
        ntrain = 8,
        ntest = -1,
        path = "../data/ModelNet10/"
    )
    # ##test
    #     for pic, multi, sample_points, y in dataloder_train :
    #         print(pic.shape, multi, sample_points.shape, y, sep="\n")
    # ##test
    diffusionModel = DiffusionModel.BaseLine(
        image_resolution = 300,
        n_gene = 1,
        device = device,
        low_threshold = 5,
        high_threshold = 10,
        bg_threshold = 0.4,
        exttype = "depth"
    )

    cloud2textFeatureModel = TransformerModel.TransformerModel(
        npoints = 2048,
        nclass = len(classes),
        fromPretrainPath = "../pretrain_transf/pointtransformer_shapenet37.pt"
    )

    imageMLP, textMLP = None, None
    classificationFn = ClassificationFn(multi, learnable = True, ahead_cat = False, device = device)

    model = MainModel(cloud2textFeatureModel, diffusionModel, clipmodel, preprocess, classificationFn, classes, (imageMLP, textMLP), meta_net=True, ahead_cat=False, multi = multi).to(device)
    if loadPath is not None :
        model.linear_textfeature = torch.load("train13-4_MainModel_linear_textfeature.pt")
        # model.linear_textfeature = torch.load(loadPath)
        model.classificationHead.pool = torch.load("train13-4_MainModel_classficationHead.pt")
    print(model.linear_textfeature)
    print(model.classificationHead.pool)
        # model.linear_textfeature = torch.load("MainModel_linear_textfeature_best.pt")
    # torch.save(model.linear_textfeature, "MainModel_linear_test.pt")
    # torch.save(model, "MainModel_all.pt")

    lossfn = torch.nn.CrossEntropyLoss() 
    opt = torch.optim.Adam([
        {'params': model.linear_textfeature.parameters(), 'lr': 1e-5},
        {'params': model.classificationHead.pool.parameters(), 'lr': 1e-5}
        # {'params': model.classificationHead.pool.parameters(), 'lr': 1e-3}
    ],eps=1e-7)
    # print(opt.param_groups[]) 

    epochs, best = 10, 0
    train_log, eval_log = [], []
    trainname = "???"
    for epoch in range(epochs) :
        def run(dataloader, mode = "train") :
            acc, tot, tloss = 0, 0, 0
            iter = tqdm(dataloader)
            for pic, multi, sample_points, y in iter :
                if pic is None : continue
                pred = model(pic, sample_points, multi)
                loss = lossfn(pred, torch.tensor(y, device = device))

                if mode == "train" :
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                guess = pred.argmax(-1).item()
                acc += (guess == y)
                tot += 1
                tloss += loss.item()
                iter.set_description("%s {epoch: %2d, acc: %.4f, loss: %5.5f}"%(mode, epoch, acc/tot, tloss/tot))

                # with open("./log/4-17-train.log", "a") as f :
                #     f.write("%d %d %.6f\n"%(guess,y,loss.item()))
                print(mode, guess, y, loss.item())
                with open("./%s.log"%trainname, "a") as f: 
                    f.write("%s %d %d %.5f %.5f %.5f\n"%(mode, guess, y, loss.item(), acc/tot, tloss/tot))
                # print((model.linear_textfeature.weight**2).sum())
                # print(model.linear_textfeature.weight)
                # print(model.linear_textfeature.weight.grad)
                # print("grad", (model.linear_textfeature.weight.grad**2).sum())
                # break
            with open("./%s+.log"%trainname, "a") as f: 
                f.write("%s %.5f %.5f\n"%(mode, acc/tot, tloss/tot))
            iter.close()
            return (acc/tot, tloss/tot)

        with torch.no_grad() :
            model.eval()
            eval_log.append(run(dataloader_test, "eval"))

        print(eval_log)
                

        model.train()
        train_log.append(run(dataloader_train, "train"))
        torch.save(model.linear_textfeature, "%s_MainModel_linear_textfeature.pt"%trainname)
        torch.save(model.classificationHead.pool, "%s_MainModel_classficationHead.pt"%trainname)
        print(train_log)
        

if __name__ == '__main__':
    import torch.nn.functional as F
    import sys, itertools
    sys.path.append("../")
    sys.path.append("../pretrain_transf")
    sys.path.append("../ControlNet/ControlNetmain/")
    sys.path.append("../Point_Transformers_master")
    import torch
    cudaid = 0
    torch.cuda.set_device(cudaid)
    device = "cuda:%d"%cudaid

    import numpy, os, cv2, sys, random, utils
    from CLIP.clip import clip
    from PIL import Image  
    from Models.MainModel import MainModel
    from MyFunctional.classificationHead import CosineSim
    from Models import Cloud2textFeatureModel, ProjectionModel, DiffusionModel
    from pretrain_transf import TransformerModel
    from pytorch_lightning import seed_everything
    from Models.ClassificationFn import ClassificationFn
    from Models.MLP import MLP
    
    seed_everything(20230717)

##
    train()
    sys.exit()
##

    # projectionModel = ProjectionModel.Perspective4view(
    #     image_resolution = 512,
    #     rotate = (-0, -35, -135)
    # )    
    projectionModel = ProjectionModel.Perspective4view_plus_autodensity(
        image_resolution = 512,
        rotate = (-0, -35, -135),
        npoints=2048
    )    
    diffusionModel = DiffusionModel.BaseLine(
        image_resolution = 512,
        n_gene = 1,
        device = device,
        low_threshold = 5,
        high_threshold = 10
    )
    with torch.no_grad():
        test(device, projectionModel, diffusionModel)
    # with torch.no_grad():
    #     run(device, projectionModel, diffusionModel)
    # with torch.no_grad():
    #     run_multi_view(device, projectionModel, diffusionModel)
    # with torch.no_grad():
    #     run_flq(device, projectionModel, diffusionModel)
