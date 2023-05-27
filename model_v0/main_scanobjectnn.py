import numpy
from PIL import Image  
from tqdm import tqdm, trange
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



def make_itself_density(data, k = 4, step = 10) :
        data = (data - data.min())
        data = data / data.max()
        data = data*20 - 10
        alldata = [data]
        planeList = []
        # print("len", len(data))
        n = len(data)
            
        # for i in range(3) :
        #     for j in range(3) :
        #         for k in range(3) :
        #             alldata.append(data+numpy.array([i/20, j/20, k/20]))
        for i in trange(n) :
            import random
            if random.random()<max(0, (n-20000)/100000) :continue
            dis = ((data - data[i])**2).sum(-1)
            arg_mink = dis.argpartition(k)[:k]
            for v1 in arg_mink :
                for v2 in arg_mink :
                    if v1!=v2 :
                        planeList.append([i, v1, v2])
            mink = data[arg_mink]

            vec_mink = mink - data[i]

            for s in range(step) :
                rate = s/step
                alldata.append(data[i] + rate * vec_mink)
        return numpy.concatenate(alldata), planeList

def one_view(dataloader):
    clipmodel, preprocess = clip.load("ViT-B-16", device=device,download_root = "../../encoder_model")
    text = ["a photo of a %s, whiter is closer" %_ for _ in classes]
    texte = clip.tokenize(["a photo of a %s" %_ for _ in classes]).to(device)
    # projectionModel = ProjectionModel.Perspective4view(
    #     image_resolution = 512,
    #     rotate = (-0,-20, -20),
    #     areadensity = 0.0003,
    # )    
    # projectionModel = ProjectionModel.Perspective(
    #     image_resolution = 512,
    #     rotate = (-0, -35, -135),
    #     areadensity = 0.0003,
    # )    
    # diffusionModel = DiffusionModel.BaseLine(
    #     image_resolution = 512,
    #     n_gene = 1,
    #     device = device,
    #     low_threshold = 5,
    #     high_threshold = 10,
    #     bg_threshold = 0.4,
    #     exttype = "depth_origin"
    # )
    for pic, multi, y in dataloader :
        print(y)
    return
    for C in range(len(classes)):
        acc, tot, counter = 0, 0, 0
        for j in range(len(test_list[C])):
            if counter>=10:
                break
            counter+=1
            file = numpy.fromfile('../data/object_dataset/%s/%s'%(classes[C], test_list[C][j]), dtype=numpy.float32)
            n = int(file[0])
            data = file[1:].reshape(-1, 11)[:, :3]
            std_data = numpy.zeros_like(data)
            std_data[:, 0] = data[:, 2]
            std_data[:, 1] = data[:, 0]
            std_data[:, 2] = data[:, 1]
            data = std_data

            data, planeList = make_itself_density(data)

            # pointList = numpy.array(f['data'])[0]
            pointList = data
            # planeList = []

            pic, multi = projectionModel(numpy.array(pointList), planeList)
            
            img = Image.fromarray(pic[0, ...].squeeze())
            img.save("./flq/test1/%s+%s'0-project-example.png"%(classes[C],test_list[C][j]))
            img = Image.fromarray(pic[1, ...].squeeze())
            img.save("./flq/test1/%s+%s'1-project-example.png"%(classes[C],test_list[C][j]))
            img = Image.fromarray(pic[2, ...].squeeze())
            img.save("./flq/test1/%s+%s'2-project-example.png"%(classes[C],test_list[C][j]))
            img = Image.fromarray(pic[3, ...].squeeze())
            img.save("./flq/test1/%s+%s'3-project-example.png"%(classes[C],test_list[C][j]))
            continue
            std_pic = pic[1, ...]
                
            x = numpy.stack([std_pic]*len(classes))
            #10 * 512 * 512 * 1
            # text = ["%s"%_ for _ in classes]
        
            ls = []
            for i in range(0, len(text), 5) :
                __, x_samples, detected_map = diffusionModel(x[i: i+5], text[i: i+5], 0, 0)
                ls.append(x_samples[0, ...]) # 1 * 5 * 512 * 512 * 3
            x_samples = torch.cat(ls, dim=0)
            
            image = x_samples  # 10 * 512 * 512 * 3

            ls = []
            for i in range(image.shape[0]) :
                ls.append(
                    preprocess(
                        Image.fromarray(
                            image[i].cpu().numpy().astype(numpy.uint8)
                            # std_pic.squeeze().astype(numpy.uint8)
                        )
                    )
                )
            image = torch.stack(ls).to(device)
            logits_per_image, logits_per_text = clipmodel(image, texte)
            probs = logits_per_image.softmax(dim=-1)
            
            ## try1 get diag max
            # guess = torch.diag(probs).argmax().item()
            
            ## try2 diag max + col sum
            # colsum=(torch.sum(probs,dim=0)-torch.diag(probs)).softmax(dim=-1)
            # colsum = (torch.sum(probs,dim=0)-torch.max(probs,dim=0)[0]*(torch.max(probs,dim=0)[1]!=torch.arange(0,len(probs)).to(device))).softmax(dim=-1)
            
            colsum = torch.sum(probs*(probs<=torch.diag(probs)),dim=0).softmax(dim=-1)
            guess = (0.4*colsum+0.6*torch.diag(probs)).argmax().item()
            tot+=1
            acc += (guess == C)
            with open("./flq/log/%s.log"%filename, "a") as f :
                # print(guess,acc,tot,C)
                # print(classes[guess])
                f.write(str(acc / tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + test_list[C][j])
                f.write("\n")
                if (guess != C) : 
                #     # debug_img = Image.fromarray(image[C, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                #     # debug_img.save("./flq/%s_C.png" %file)
                #     # debug_img = Image.fromarray(image[guess, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                #     # debug_img.save("./flq/%s_G.png" %file)
                #     # f.write(str((log_probs.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).cpu().numpy().tolist()))
                    f.write(str(colsum.cpu().numpy().tolist()))
                    f.write("\n\n")
                    tmp_put=probs.cpu().numpy().tolist()
                    for i in range(probs.shape[0]):
                        f.write(str(tmp_put[i]))
                        f.write(str("\n"))
                #     # f.write(str(probs.cpu().numpy().tolist()))
                #     f.write("\n")
            print(acc / (tot), tot, "ans=", classes[C], "guess=", classes[guess])
            # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        # with open("./flq/log/%s+.log"%filename, "a") as f:
        #     f.write(str(acc/(tot))+" "+str((tot))+" a="+classes[C] )
        #     f.write("\n")

def multi_view(dataloader,filename, classes):
    clipmodel, preprocess = clip.load("ViT-B-16", device=device,download_root = "../../encoder_model")   
    diffusionModel = DiffusionModel.BaseLine(
        image_resolution = 512,
        n_gene = 1,
        device = device,
        low_threshold = 5,
        high_threshold = 10,
        bg_threshold = 0.4,
        exttype = "depth_origin",
        detect_resolution=225
    )
    lstC = -1
    acc, tot = 0, 0
    
    skiptool = 0
    from tqdm import tqdm
    for pic,multi,C in tqdm(dataloader):

        if multi == None : continue
    
        std_pic = pic[:, ...]# 4 * 512 * 512  *1
        x = numpy.concatenate([std_pic]*len(classes), axis=0)# (C * 4) * 512 * 512  *1
        #10 * 512 * 512 * 1
        text = list(itertools.chain(*[["%s, behind a building"%_]*multi for _ in classes]))
        ls = []
        for i in range(0, len(text), 5) :
            __, x_samples, detected_map = diffusionModel(x[i: i+5], text[i: i+5], 0, 0)
            ls.append(x_samples[0, ...]) # 1 * 5 * 512 * 512 * 3
        x_samples = torch.cat(ls, dim=0)
        
        image = x_samples  # (C * 10 )* 512 * 512 * 3

        
        # debug_img = Image.fromarray(image[0, ...].squeeze().cpu().numpy().astype(numpy.uint8))
        # debug_img.save("./debug/tmpbed.png")
        # debug_img = Image.fromarray(image[1, ...].squeeze().cpu().numpy().astype(numpy.uint8))
        # debug_img.save("./debug/tmpbath.png")
        # return 
        # : check passed

        text = ["a 3D image of a %s, maybe some of it is blocked by something else" %_ for _ in classes]
        texte = clip.tokenize(text).to(device)

        ls = []
        for i in range(image.shape[0]) :
            ls.append(
                preprocess(
                    Image.fromarray(
                        image[i].cpu().numpy().astype(numpy.uint8)
                        # std_pic.squeeze().astype(numpy.uint8)
                    )
                )
            )
        image = torch.stack(ls).to(device)

        # debug_img = Image.fromarray(image[0, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
        # debug_img.save("./debug/tmpbed.png")
        # debug_img = Image.fromarray(image[1, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
        # debug_img.save("./debug/tmpbath.png")
        # return 

        # image = image.transpose(1,2).transpose(2,3)# 10 * 3 * 224 * 224
        
        image_features = clipmodel.encode_image(image)
        text_features = clipmodel.encode_text(texte)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        print("image_features:",image_features.shape)
        print("text_features:",text_features.shape)
        # ###deal image_features 4 to 1
        # image_features = torch.nn.functional.avg_pool2d(
        #     image_features[None, ...], (multi, 1)
        # )[0, ...]
        # print(text_features.shape)
        # print(text_features.shape)
        ###
        logit_scale = clipmodel.logit_scale.exp()
        mylogits_per_image = logit_scale * image_features @ text_features.t()
        # print("mylogits_per_image:",mylogits_per_image.shape)
        # ###deal image_Prop 4 to 1
        mylogits_per_image = torch.nn.functional.max_pool2d(
            mylogits_per_image[None, ...], (multi, 1)
        )[0, ...]
        # print(text_features.shape)
        # print(text_features.shape) 
        ###
        probs = mylogits_per_image.softmax(dim=-1)
        

        # logits_per_image, logits_per_text = clipmodel(image, texte)
        # probs = logits_per_image.softmax(dim=-1)

##old for v2
        log_probs = torch.log(probs)
        topers = (log_probs.sort(dim=0)[0])[:, :]
        global_info = topers.mean(dim=0)
        global_info = torch.exp(global_info)
        # global_info = global_info.softmax(dim=-1)
        global_info -= global_info.min()
        global_info /= global_info.max()
        guess = (global_info*probs.max(dim=0)[0]).argmax().item()
##
##new for v3
        # colsum = torch.sum(probs*(probs<=torch.diag(probs)),dim=0).softmax(dim=-1)
        # guess = (0.4*colsum+0.6*torch.diag(probs)).argmax().item()
##


        # print(topers.sum(dim=0).softmax(dim=-1))
        # print(probs.max(dim=0)[0])
        # guess = (probs.max(dim=0)[0]).argmax().item()
        
        if lstC != C :
            if lstC != -1 :
                with open("./flq/log/%s+.log"%filename, "a") as f:
                    f.write(str(acc/tot)+" "+str(tot)+" a="+classes[lstC])
                    f.write("\n")
                acc, tot = 0, 0
            lstC = C
        acc += (guess == C)
        tot += 1

        with open("./flq/log/%s.log"%filename, "a") as f :
            f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess])
            f.write("\n")
            # if (guess != C) : 
            #     f.write(str((log_probs.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).cpu().numpy().tolist()))
            #     f.write("\n")
        print(acc / tot, tot, "ans=", classes[C], "guess=", classes[guess])
        # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        

from torch.utils.data import Dataset, DataLoader

class ScanobjDataset(Dataset) :
    def __init__(self, classes, guide_file, projectionModel, path, testNumber=-1) :
        super(ScanobjDataset, self).__init__()
        test_list = [[] for i in range(len(classes))]
        with open(guide_file) as f:
            for line in f:
                l=line.strip().split('\t')
                if l[-1]=='t':
                    test_list[int(l[1])].append(l[0])
        
        self.x = []
        self.y = []
        for i in range(len(classes)) :
            for j in range(len(test_list[i])) :
                filename = test_list[i][j]
                self.x.append((classes[i], filename, j))
                self.y.append(i)
        print(self.x)
                
        self.len = len(self.y)
        
        self.model = projectionModel

        self.testNumber = testNumber

        self.path_format = path + '/%s/%s'  

        self.skiptool = 0 
            
    def __getitem__(self, id):
        if self.testNumber>=0 and self.x[id][2] >= self.testNumber : return None, None, None
        file = numpy.fromfile(self.path_format%(self.x[id][0], self.x[id][1]), dtype=numpy.float32)
        n = int(file[0])
        data = file[1:].reshape(-1, 11)[:, :3]
        std_data = numpy.zeros_like(data)
        std_data[:, 0] = data[:, 2]
        std_data[:, 1] = data[:, 0]
        std_data[:, 2] = data[:, 1]
        data = std_data

        data, planeList = make_itself_density(data)

        # pointList = numpy.array(f['data'])[0]
        pointList = data
        # planeList = []

        pic, multi = self.model(numpy.array(pointList), planeList)
        
        # img = Image.fromarray(pic[0, ...].squeeze())
        # img.save("./flq/test3/%s+%s'0-project-example.png"%(classes[self.y[id]],self.x[id][1]))
        # img = Image.fromarray(pic[1, ...].squeeze())
        # img.save("./flq/test3/%s+%s'1-project-example.png"%(classes[self.y[id]],self.x[id][1]))
        # img = Image.fromarray(pic[2, ...].squeeze())
        # img.save("./flq/test3/%s+%s'2-project-example.png"%(classes[self.y[id]],self.x[id][1]))
        # img = Image.fromarray(pic[3, ...].squeeze())
        # img.save("./flq/test3/%s+%s'3-project-example.png"%(classes[self.y[id]],self.x[id][1]))
        
        return pic, multi, self.y[id]

    def __len__(self):
        return self.len

class DataLoaderX(DataLoader):
    def __iter__(self):
        from prefetch_generator import BackgroundGenerator
        return BackgroundGenerator(super().__iter__())

def collate_fn(x) :
    pic, multi, y = x[0]
    return pic, multi, y

def getDataloader(classes, guide_file, projectionModel, path = "../data/object_dataset") :
    dataset = ScanobjDataset(classes, guide_file, projectionModel, path,testNumber=-1)
    dataloder = DataLoaderX(dataset,collate_fn=collate_fn,shuffle=False,num_workers = 2)
    return dataloder
    
if __name__ =='__main__':
    import threading
    mx = threading.Lock()

    seed_everything(1136514422)
    
    classes = ["bag","bin","box","cabinet","chair","desk","display","door","shelf","table","bed","pillow","sink","sofa","toilet"]
    projectionModel = ProjectionModel.Perspective4view(
        image_resolution = 512,
        rotate = (-0,-20, -25),
        areadensity = 0.0003,
    )    
    guide_file = "../data/object_dataset/split_new.txt"
    dataloader = getDataloader(classes, guide_file, projectionModel)
    
    filename='4vScanV16_test_all_v5_+common_CLIPprompt_+flq_predict_func_+sota_CLIPprompt_+my_CLIPprompt_-flq_predict_func'
    #sotaCLIPprompt = with xxx
    #myCLIPprompt = a 3D image of a %s, maybe some of it is blocked by something else
    
    
    # with torch.no_grad():
    #     one_view(dataloader)
    with torch.no_grad():
        multi_view(dataloader,filename=filename, classes = classes)
