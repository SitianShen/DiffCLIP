import torch, numpy, itertools, math
from PIL import Image
from MyFunctional.classificationHead import CosineSim
from CLIP.clip import clip
from tqdm import tqdm

class MainModel(torch.nn.Module) :
    def __init__(self, cloud2textFeatureModel, diffusionModel, clipModel, clippreprocess, classificationHead, classes, ITMLP = None, meta_net = False, ahead_cat=False, multi=None, device = "cuda:0") :
        super(MainModel, self).__init__()
        self.cloud2textFeatureModel = cloud2textFeatureModel
        # self.projectionModel = projectionModel # may support more than one output(project from various angle)
        self.diffusionModel = diffusionModel
        self.clipmodel = clipModel
        self.clippreprocess = clippreprocess
        self.classificationHead = classificationHead
        self.classes = classes
        self.device = device
        self.meta_net = meta_net
        self.ahead_cat = ahead_cat
    
        # self.linear_textfeature = torch.nn.ModuleList(
        #     [
        #         torch.nn.Linear(
        #             32 * 2 ** cloud2textFeatureModel.nblocks, clipModel.transformer_width, device = self.device
        #         )
        #         for i in range(len(classes))
        #     ]
        # )
        if (meta_net == False) :
            self.linear_textfeature = torch.nn.Linear(32 * 2 ** cloud2textFeatureModel.nblocks, clipModel.transformer_width, device = self.device)
        else :
            nbs = cloud2textFeatureModel.nblocks
            wth = clipModel.transformer_width
            self.linear_textfeature = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(32*2**i, int(math.sqrt(32*2**i*wth))),
                    torch.nn.ReLU(),
                    torch.nn.Linear(int(math.sqrt(32*2**i*wth)), wth)
                ) for i in range(nbs+1)
            ]).to(device)

        if ITMLP is not None :
            self.imageMLP, self.textMLP = ITMLP
        else :
            self.imageMLP, self.textMLP = None, None

        if ahead_cat == True :
            in_c = clipModel.transformer_width*multi
            out_c = clipModel.transformer_width
            # if ITMLP is not None :
            #     if ITMLP[0] is not None :
            #         in_c*=2
            #         out_c*=2
            self.multiview_net = torch.nn.Sequential(
                torch.nn.Linear(in_c, int((in_c*out_c)**(2/3))),
                torch.nn.ReLU(),
                torch.nn.Linear(int((in_c*out_c)**(2/3)),int((in_c*out_c)**(1/3))),
                torch.nn.ReLU(),
                torch.nn.Linear(int((in_c*out_c)**(1/3)), out_c)
            )
            self.multiview_net_in_c = in_c
        else :
            self.multiview_net = None
        
    # def embedding(text, tokenize, embedding) :
    #     aux = "%s" # "a picture of %s"
    #     return embedding(tokenize(aux % text))
    
    # def makePrompt(textExtraPrompt, textStdPrompt) :
    #     if (textExtraPrompt is None) :
    #         return textStdPrompt
    #     return torch.stack([textExtraPrompt, textStdPrompt], dim = -1) # bs * dim
        
    def forward(self, img, cloud3d, multi) :
        x = numpy.concatenate([img]*len(self.classes), axis=0)
        text = list(itertools.chain(*[["%s"%_]*multi for _ in self.classes]))

        ls = []
        stepl = 10
        # for i in tqdm(range(0, len(text), stepl), leave = False) :
        for i in range(0, len(text), stepl) :
            __, x_samples, detected_map = self.diffusionModel(x[i: i+stepl], text[i: i+stepl], 0, 0)
            ls.append(x_samples[0, ...]) # 1 * 5 * 512 * 512 * 3
        x_samples = torch.cat(ls, dim=0)
        # x_samples = torch.cat(

        #     dim=0
        # )

        image = x_samples

        text = ["%s" %_ for _ in self.classes]
        texte = clip.tokenize(text).to(self.device)

        ls = []
        for i in range(image.shape[0]) :
            ls.append(
                self.clippreprocess(
                    Image.fromarray(
                        image[i].cpu().numpy().astype(numpy.uint8)
                        # std_pic.squeeze().astype(numpy.uint8)
                    )
                )
            )
        image = torch.stack(ls).to(self.device)

        lst_feature, xyz_and_features = self.cloud2textFeatureModel.backbone(torch.from_numpy(cloud3d).to(self.device))
        features = list(zip(*xyz_and_features))[1]
        # print(ext_feature.mean(1)[0, ...])
        if self.meta_net == False :
            ext_feature = self.linear_textfeature(lst_feature.mean(1))
            # ext_feature.sum().backward()
            text_features = self.clipmodel.encode_text_extfeature(texte, ext_feature)
        else :
            ext_feature = torch.cat([
                self.linear_textfeature[i](features[i].mean(1)) for i in range(len(features))
            ], dim=0)
            text_features = self.clipmodel.encode_text_extfeature(texte, ext_feature)


        image_features = self.clipmodel.encode_image(image)

        # text_features.sum().backward()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        image_features = image_features.to(torch.float32)
        text_features = text_features.to(torch.float32)
        
        if self.imageMLP is not None :
            # image_features = torch.cat([self.imageMLP(image_features), image_features], dim=-1)
            image_features = self.imageMLP(image_features)

        if self.textMLP is not None :
            # text_features = torch.cat([self.textMLP(text_features), text_features], dim=-1)
            text_features = self.textMLP(text_features)
            
        print("igf", image_features)
        print("txf", text_features)

        if self.ahead_cat == True :
            image_features = self.multiview_net(image_features.reshape(-1, self.multiview_net_in_c))

        logit_scale = self.clipmodel.logit_scale.exp()
        mylogits_per_image = logit_scale * image_features @ text_features.t()
        mylogits_per_image/=mylogits_per_image.sum()
        mylogits_per_image*=mylogits_per_image.shape[0]*mylogits_per_image.shape[1]
        # ###deal image_Prop 4 to 1
        print("mls", mylogits_per_image)
        pred = self.classificationHead(mylogits_per_image)
        print("pred", pred)
        return pred


        # textExtraPrompt = self.cloud2textFeatureModel(cloud3d)
        # textStdPrompt = self.embedding(text, clip.tokenize, self.clip.token_embedding)
        # textPrompt, dimLen = self.makePrompt(textExtraPrompt, textStdPrompt)
        
        # projected_pic2d, multi = self.projectionModel(cloud3d, self.ismultiProjection) # may be a list of pic from diff angle
        # diffusion_pic2d = self.diffusionModel(projected_pic2d, [text]*multi, textExtraPrompt, self.ismultiProjection) # same
        
        # if (self.ismultiProjection == False) : #if 'diffusion_pic2d' is a list (from 6 angles)
        #     image_feature = self.clip.encode_image(diffusion_pic2d)
        #     text_feature  = self.clip.encode_text_skipEmbedding(textPrompt, dimLen)
        
        #     pred = self.classificationHead(image_feature, text_feature)
        #     return pred
        # else :
        #     raise NotImplementedError

if __name__ == '__main__':
    import sys
    sys.path.append("../../")