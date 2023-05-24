import torch, numpy, einops, random
from ControlNet.ControlNetmain.annotator.canny import CannyDetector
from pytorch_lightning import seed_everything
from ControlNet.ControlNetmain.cldm.model import create_model, load_state_dict
from ControlNet.ControlNetmain.cldm.ddim_hacked import DDIMSampler

picSize = 224
class BaseLine(torch.nn.Module) :
    def __init__(self, image_resolution, n_gene: int, device = "cuda:0", exttype = "canny", seed = -1, low_threshold = 50, high_threshold = 100) :
        super(BaseLine, self).__init__()
        self.image_resolution = image_resolution
        self.apply_canny = CannyDetector()
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
        
        self.model = create_model('../ControlNet/ControlNetmain/models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('../ControlNet/ControlNetmain/models/control_sd15_canny.pth', location="cpu"))
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
        print("image.shape", image.shape)
        batch_size, H, W, C = image.shape
        
        if (self.extra_type == "canny") :
            ls = []
            for i in range(batch_size) :
                ls.append(self.apply_canny(image[i], self.low_threshold, self.high_threshold))
            detected_map = numpy.stack(ls)
            print("detected_map.shape", detected_map.shape)
        else :
            raise NotImplementedError

        detected_map = self.HWC3(detected_map)
        
        control = torch.from_numpy(detected_map.copy()).float().to(self.device) / 255.0
        control = torch.concat([control for _ in range(self.n_gene)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        if self.seed == -1:
            self.seed = random.randint(0, 65535)
        seed_everything(self.seed)

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