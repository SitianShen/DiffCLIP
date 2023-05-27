import torch
class ClassificationFn(torch.nn.Module) :
    def __init__(self, multi, learnable = False, ahead_cat = False, device = "cuda:0") :
        super(ClassificationFn, self).__init__()
        self.multi = multi
        self.learnable = learnable
        if learnable == True :
            self.pool = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, (multi, 1), (multi, 1), device = device),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 16, (1, 1), (1, 1), device = device),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 4, (1, 1), (1, 1), device = device),
                torch.nn.Conv2d(4, 1, (1, 1), (1, 1), device = device, bias = False)
            )
        else :
            self.pool = torch.nn.functional.max_pool2d
        self.ahead_cat = ahead_cat

    def forward(self, mylogits_per_image) :
        if self.ahead_cat == False :
            if self.learnable == True :
                mylogits_per_image = self.pool(
                    mylogits_per_image[None, ...]
                )[0, ...]
            else :
                mylogits_per_image = torch.nn.functional.max_pool2d(
                    mylogits_per_image[None, ...], (self.multi, 1)
                )[0, ...]
        else : 
            pass


        probs = mylogits_per_image.softmax(dim=-1)
        # # probs = mylogits_per_image
        
        log_probs = torch.log(probs+1e-20)
        topers = (log_probs.sort(dim=0)[0])[:, :]
        global_info = topers.mean(dim=0)
        expglobal_info = torch.exp(global_info)
        # global_info = global_info.softmax(dim=-1)
        expglobal_info1 = expglobal_info - expglobal_info.min().detach()
        expglobal_info2 = expglobal_info1 / expglobal_info1.max().detach()

        # print(expglobal_info2)
        # print(probs.max(dim=0)[0])

        pred = expglobal_info2*probs.max(dim=0)[0]
        # pred = probs.max(dim=0)[0]

        return pred
