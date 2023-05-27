import torch, numpy, einops, random
import cv2
from ControlNet.ControlNetmain.annotator.canny import CannyDetector
from ControlNet.ControlNetmain.annotator.midas import MidasDetector
from ControlNet.ControlNetmain.annotator.hed import HEDdetector,nms
from ControlNet.ControlNetmain.annotator.midas import MidasDetector
from ControlNet.ControlNetmain.annotator.openpose import OpenposeDetector
from ControlNet.ControlNetmain.annotator.uniformer import UniformerDetector
from ControlNet.ControlNetmain.cldm.model import create_model, load_state_dict
from ControlNet.ControlNetmain.annotator.mlsd import MLSDdetector
from ControlNet.ControlNetmain.cldm.ddim_hacked import DDIMSampler
from ControlNet.ControlNetmain.annotator.util import resize_image
from pytorch_lightning import seed_everything
picSize = 224
class BaseLine(torch.nn.Module) :
    def __init__(self, image_resolution, n_gene: int, device = "cuda:0", exttype = "canny", seed = -1, low_threshold = 50, high_threshold = 100,bg_threshold = 0.4,value_threshold=0.1,distance_threshold=0.1,detect_resolution=225) :
        super(BaseLine, self).__init__()
        self.image_resolution = image_resolution
        self.detect_resolution = detect_resolution
        self.device = device
        self.n_gene = n_gene
        
        self.add_prompt = "best quality, extremely detailed"
        self.guess_mode = False
        self.n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        self.strength = 1.
        self.ddim_steps = 20
        self.eta = 0
        self.scale = 9.
        self.extra_type = exttype
        self.seed = seed
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.bg_threshold = bg_threshold
        self.value_threshold = value_threshold
        self.distance_threshold =distance_threshold
        
        if exttype == "canny":
            self.gradio = CannyDetector()
        elif exttype == "depth" or exttype == "depth_origin":
            exttype = "depth"
            self.gradio = MidasDetector()
        elif exttype == "hed":
            self.gradio = HEDdetector()
        elif exttype == "normal":
            self.gradio = MidasDetector()
        elif exttype == "scribble":
            self.gradio = ''
        elif exttype == "fake_scribble":
            exttype = "scribble"
            self.gradio = HEDdetector()
        elif exttype == "openpose":
            self.gradio = OpenposeDetector()
        elif exttype == "seg":
            self.gradio = UniformerDetector()
        elif exttype == "mlsd":
            self.gradio = MLSDdetector()
        else :
            raise NotImplementedError
        
        self.model = create_model('../ControlNet/ControlNetmain/models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('../ControlNet/ControlNetmain/models/control_sd15_%s.pth'%exttype, location="cpu"))
        self.model = self.model.to(self.device)
        self.ddim_sampler = DDIMSampler(self.model)
   
    
    def HWC3(self, x):
        # print(x.shape)
        if x.ndim == 3:
            x = x[:, :, :, None]
        assert x.ndim == 4
        bs, H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return numpy.concatenate([x, x, x], axis=-1)
        # if C == 4:
        #     color = x[:, :, 0:3].astype(np.float32)
        #     alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        #     y = color * alpha + 255.0 * (1.0 - alpha)
        #     y = y.clip(0, 255).astype(np.uint8)
        #     return y
    def forward(self, projected_pic2d: numpy, text: list, textExtraPrompt, ismultiProjection) :
        # image = resize_image(self.HWC3(projected_pic2d), self.image_resolution) #bs*h*w*c
        image = self.HWC3(projected_pic2d)
        # print("image.shape:", image.shape)
        batch_size, H, W, C = image.shape
        
        ls = []
        if (self.extra_type == "canny") :
            for i in range(batch_size) :
                ls.append(self.gradio(image[i], self.low_threshold, self.high_threshold))
        elif (self.extra_type == "depth_origin"):
            for i in range(batch_size):
                detected_tmp,_=self.gradio(resize_image(image[i],self.detect_resolution))
                ls.append(detected_tmp)
        elif (self.extra_type == 'depth'):
            for i in range(batch_size):
                # detected_tmp,_=self.gradio(resize_image(image[i],self.detect_resolution))
                # ls.append(detected_tmp)
                ls.append(image[i])
        elif (self.extra_type == "hed" or self.extra_type == "fake_scribble" or self.extra_type == "seg"):
            for i in range(batch_size):
                ls.append(self.gradio(image[i]))
        elif (self.extra_type == "normal"):
            for i in range(batch_size):
                _,detected_tmp=self.gradio(image[i],bg_th=self.bg_threshold)
                ls.append(detected_tmp)
        elif (self.extra_type == "scribble"):
            ls = numpy.zeros_like(image, dtype=numpy.uint8)
            ls[numpy.min(image, axis=3) < 127] = 255        
        elif (self.extra_type == "mlsd"):
            for i in range(batch_size):
                ls.append(self.gradio(image[i],self.value_threshold, self.distance_threshold))
        else :
            raise NotImplementedError
        detected_map = numpy.stack(ls)
        detected_map = self.HWC3(detected_map)
        if detected_map.shape[1]!=H:
            detected_new = numpy.zeros_like(image)
            for i in range(batch_size):
                # print("det:",detected_map[i].shape)#(768,768,3)
                # print(detected_map[i])
                detected_new[i] = cv2.resize(detected_map[i], (W, H), interpolation=cv2.INTER_LINEAR)
            detected_map = detected_new
        # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        # print("detected_map.shape:", detected_map.shape)#(B,H,W,3)
        
        if (self.extra_type == "fake_scribble"):
            for i in range(batch_size): 
                detected_map[i] = nms(detected_map[i], 127, 3.0)
                detected_map[i] = cv2.GaussianBlur(detected_map[i], (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0
        
        control = torch.from_numpy(detected_map.copy()).float().to(self.device) / 255.0
        control = torch.concat([control for _ in range(self.n_gene)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        # if self.seed == -1:
        #     seed = random.randint(0, 65535)
        # seed_everything(seed)
        
        text = list(map(lambda s : s + "," + self.add_prompt, text))
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning(text * self.n_gene)]}
        un_cond = {"c_concat": None if self.guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([self.n_prompt] * self.n_gene * len(text))]}
        
        shape = (4, H // 8, W // 8)
        
        self.model.control_scales = [self.strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([self.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = self.ddim_sampler.sample(self.ddim_steps, self.n_gene * len(text),
                                                     shape, cond, verbose=False, eta=self.eta,
                                                     unconditional_guidance_scale=self.scale,
                                                     unconditional_conditioning=un_cond)
        
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).clip(0, 255)

        results_image = [x_samples[i*batch_size:(i+1)*batch_size].cpu().numpy().astype(numpy.uint8) for i in range(self.n_gene)]
        
        shape = x_samples.shape
        x_samples = x_samples.reshape(self.n_gene, batch_size, shape[1], shape[2], shape[3])
        
        return results_image, x_samples, detected_map     # 