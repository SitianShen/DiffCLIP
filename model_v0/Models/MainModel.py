import torch
from MyFunctional.classificationHead import CosineSim
from CLIP.clip import clip

class MainModel(torch.nn.Module) :
    def __init__(self, cloud2textFeatureModel, projectionModel, diffusionModel, clipModel, classificationHead = CosineSim, ismultiProjection = False) :
        super(MainModel, self).__init__()
        self.cloud2textFeatureModel = cloud2textFeatureModel
        self.projectionModel = projectionModel # may support more than one output(project from various angle)
        self.diffusionModel = diffusionModel
        self.clip = clipModel
        self.classificationHead = classificationHead
        self.ismultiProjection = ismultiProjection
        
    def embedding(text, tokenize, embedding) :
        aux = "%s" # "a picture of %s"
        return embedding(tokenize(aux % text))
    
    def makePrompt(textExtraPrompt, textStdPrompt) :
        if (textExtraPrompt is None) :
            return textStdPrompt
        return torch.stack([textExtraPrompt, textStdPrompt], dim = -1) # bs * dim
        
    def forward(self, cloud3d, text) :
        textExtraPrompt = self.cloud2textFeatureModel(cloud3d)
        textStdPrompt = self.embedding(text, clip.tokenize, self.clip.token_embedding)
        textPrompt, dimLen = self.makePrompt(textExtraPrompt, textStdPrompt)
        
        projected_pic2d, multi = self.projectionModel(cloud3d, self.ismultiProjection) # may be a list of pic from diff angle
        diffusion_pic2d = self.diffusionModel(projected_pic2d, [text]*multi, textExtraPrompt, self.ismultiProjection) # same
        
        if (self.ismultiProjection == False) : #if 'diffusion_pic2d' is a list (from 6 angles)
            image_feature = self.clip.encode_image(diffusion_pic2d)
            text_feature  = self.clip.encode_text_skipEmbedding(textPrompt, dimLen)
        
            pred = self.classificationHead(image_feature, text_feature)
            return pred
        else :
            raise NotImplementedError

if __name__ == '__main__':
    import sys
    sys.path.append("../../")