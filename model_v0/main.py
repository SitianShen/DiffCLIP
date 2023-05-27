def run_flq(device, projectionModel, diffusionModel):
    filename='1vN10R50*4'
    clipmodel, preprocess = clip.load("RN50x4", device=device,download_root = "../../encoder_model")
    
    path = "../data/ModelNet10/"
    # classes = ["nightstand", "desk","dresser", "bathtub"]
    classes = [ "monitor", "bed", "chair", "toilet", "table", "sofa", "nightstand", "dresser","desk", "bathtub"]
    # path = "../data/ModelNet40/"
    # classes = ["nightstand", "desk","dresser", "bathtub","monitor", "bed", "chair", "toilet", "table", "sofa"]
    # classes = ["nightstand", "dresser", "desk", "bathtub","monitor", "bed", "chair", "toilet", "table","sofa"]
    # classes = ['stairs', 'curtain', 'sink', 'xbox', 'plant', 'car', 'bed', 'chair', 'lamp', 'door', 'wardrobe', 'cone', 'toilet', 'bench', 'bowl', 'desk', 'airplane', 'bottle', 'mantel', 'radio', 'person', 'cup', 'bathtub', 'monitor', 'sofa', 'vase', 'laptop', 'table', 'keyboard', 'piano', 'tent', 'guitar', 'stool', 'bookshelf', 'dresser', 'TV stand', 'range hood', 'nightstand', 'glass box', 'flower pot']
    # random.shuffle(classes)
    text = ["a photo of a %s, whiter is closer" %_ for _ in classes]
    texte = clip.tokenize(["a 3D image of a %s with rendered background." %_ for _ in classes]).to(device)
    # text_features = clipmodel.encode_text(texte)#[10,512]
    # # print(text_features.shape,text_features)
    # cos=F.cosine_similarity(text_features.unsqueeze(1),text_features.unsqueeze(0),dim=-1)
    # idx=cos.sort(dim=1,descending=True)[1][:,1:4]
    # prompt_text = []
    # for i in range(idx.shape[0]):
    #     tmp_text = classes[i]+', without '
    #     for j in range(idx.shape[1]):
    #         if j+1==idx.shape[1]:
    #             tmp_text+=classes[idx[i][j]]+'.'
    #             continue
    #         tmp_text+=classes[idx[i][j]]+' and '
    #     prompt_text.append(tmp_text)
        
    # print(prompt_text)
    # prompt_texte =  clip.tokenize(prompt_text).o(device)
    
    # num_test = 20
    for C in range(len(classes)) :
        acc, tot, counter = 0, 0, 0
        full_path = path + classes[C] + "/test/"
        files = os.listdir(full_path)
        # skiprate = len(files)//num_test
        for file in files :
            if file[-1] != 'f' : continue
            counter += 1
            # print(file, counter)
            # if tot>=num_test : break
            # if (counter % skiprate != 0) :continue
         ###############test#################
            with open(full_path + file) as f:
                # assert f.readline().strip() == "OFF"
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
            pointList = numpy.array(pointList)
            pic, multi = projectionModel(pointList.copy(), planeList)
            # 3 * 512 * 512  *1
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

            
            # debug_img = Image.fromarray(image[0, ...].squeeze().cpu().numpy().astype(numpy.uint8))
            # debug_img.save("./debug/tmpbed.png")
            # debug_img = Image.fromarray(image[1, ...].squeeze().cpu().numpy().astype(numpy.uint8))
            # debug_img.save("./debug/tmpbath.png")
            # return 
            # : check passed
            ###detect for LWH
            # L,W,H=pointList.max(axis=0)-pointList.min(axis=0)
            # print("LWH:",L,W,H)
            # text = ["%s,its length is %.3f cm,its width is %.3f cm and its hight is %.3f cm"\
            #     %(_,L,W,H) for _ in classes]
            # text = ["The picture of %s,its hight is %.3f"\
            #     %(_,H) for _ in classes]
            
            # print("================================================================")
            # print(texte.shape)
            
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
            # print(image.shape)
            # debug_img = Image.fromarray(image[0, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
            # debug_img.save("./debug/tmpbed.png")
            # debug_img = Image.fromarray(image[1, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
            # debug_img.save("./debug/tmpbath.png")
            # return 

            # image = image.transpose(1,2).transpose(2,3)# 10 * 3 * 224 * 224
            
            logits_per_image, logits_per_text = clipmodel(image, texte)
            probs = logits_per_image.softmax(dim=-1)
            
            ## try1 get diag max
            # guess = torch.diag(probs).argmax().item()
            
            ## try2 diag max + col sum
            # colsum=(torch.sum(probs,dim=0)-torch.diag(probs)).softmax(dim=-1)
            # colsum = (torch.sum(probs,dim=0)-torch.max(probs,dim=0)[0]*(torch.max(probs,dim=0)[1]!=torch.arange(0,len(probs)).to(device))).softmax(dim=-1)
            
            colsum = torch.sum(probs*(probs<=torch.diag(probs)),dim=0).softmax(dim=-1)
            guess = (0.4*colsum+0.6*torch.diag(probs)).argmax().item()
            # guess = (colsum*torch.diag(probs)).argmax().item()
            # guess = (colsum*torch.diag(probs)).argmax().item()
            # print("===============")
            # for ii in range(40) :
            #     for jj in range(40) :
            #         print(probs[ii, jj].item(), end=" ")
            #     print()
            
            # log_probs = torch.log(probs)
            # topers = (log_probs.sort(dim=0)[0])[:6, :]
            # guess = (topers.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).argmax().item()
            # print(topers.sum(dim=0).softmax(dim=-1))
            # print(probs.max(dim=0)[0])
            # guess = (probs.max(dim=0)[0]).argmax().item()
            tot+=1
            acc += (guess == C)
            with open("./flq/N10prompt/%s.log"%filename, "a") as f :
                f.write(str(acc / tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
                f.write("\n")
                # if (guess != C) : 
                # #     # debug_img = Image.fromarray(image[C, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                # #     # debug_img.save("./flq/%s_C.png" %file)
                # #     # debug_img = Image.fromarray(image[guess, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                # #     # debug_img.save("./flq/%s_G.png" %file)
                # #     # f.write(str((log_probs.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).cpu().numpy().tolist()))
                #     f.write(str(colsum.cpu().numpy().tolist()))
                #     f.write("\n")
                #     tmp_put=probs.cpu().numpy().tolist()
                #     for i in range(probs.shape[0]):
                #         f.write(str(tmp_put[i]))
                #         f.write(str("\n"))
                #     # f.write(str(probs.cpu().numpy().tolist()))
                #     f.write("\n")
            print(acc / (tot), tot, "ans=", classes[C], "guess=", classes[guess])
            # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        with open("./flq/N10prompt/%s+.log"%filename, "a") as f:
            f.write(str(acc/(tot))+" "+str((tot))+" a="+classes[C] )
            f.write("\n")

def run_N40(device, projectionModel,diffusionModel):
    filename='1vN40V16_t1'
    clipmodel, preprocess = clip.load("ViT-B-16", device=device,download_root = "../../encoder_model")
    
    # path = "../data/ModelNet10/"
    # classes = ["nightstand", "desk","dresser", "bathtub"]
    # classes = [ "monitor", "bed", "chair", "toilet", "table", "sofa", "nightstand", "dresser","desk", "bathtub"]
    path = "../data/ModelNet40/"
    # classes = ["nightstand", "desk","dresser", "bathtub","monitor", "bed", "chair", "toilet", "table", "sofa"]
    # classes = ["nightstand", "dresser", "desk", "bathtub","monitor", "bed", "chair", "toilet", "table","sofa"]
    classes = ['stairs', 'curtain', 'sink', 'xbox', 'plant', 'car', 'bed', 'chair', 'lamp', 'door', 'wardrobe', 'cone', 'toilet', 'bench', 'bowl', 'desk', 'airplane', 'bottle', 'mantel', 'radio', 'person', 'cup', 'bathtub', 'monitor', 'sofa', 'vase', 'laptop', 'table', 'keyboard', 'piano', 'tent', 'guitar', 'stool', 'bookshelf', 'dresser', 'TV stand', 'range hood', 'nightstand', 'glass box', 'flower pot']
    # random.shuffle(classes)
    text = ["a photo of a %s, whiter is closer" %_ for _ in classes]
    texte = clip.tokenize(["a photo of a %s" %_ for _ in classes]).to(device)
    num_test = 10
    for C in range(33,len(classes)) :
        acc, tot, counter = 0, 0, 0
        full_path = path + classes[C] + "/test/"
        files = os.listdir(full_path)
        skiprate = len(files)//num_test
        for file in files :
            if file[-1] != 'f' : continue
            counter += 1
            print(file, counter)
            continue
            if tot>=num_test : break
            if (counter % skiprate != 0) :continue
         ###############test#################
            with open(full_path + file) as f:
                # assert f.readline().strip() == "OFF"
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
            pointList = numpy.array(pointList)
            pic, multi = projectionModel(pointList.copy(), planeList)
            # 3 * 512 * 512  *1
            std_pic = pic[1, ...]
            # print(std_pic.shape) (512,512,1)
            x=std_pic.copy()
            # x = numpy.stack([std_pic]*len(classes))

            # 10 * 512 * 512 * 1
            # text = ["%s"%_ for _ in classes]
        
            # ls = []
            # for i in range(0, len(text), 5) :
            #     __, x_samples, detected_map = diffusionModel(x[i: i+5], text[i: i+5], 0, 0)
            #     ls.append(x_samples[0, ...]) # 1 * 5 * 512 * 512 * 3
            # x_samples = torch.cat(ls, dim=0)
            x = numpy.concatenate([x, x, x], axis=-1)
            # print(x.shape)
            image = torch.from_numpy(x).to(device).unsqueeze(0)  # 1*512 * 512 * 3
            # print(image.shape)
           
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
            image = torch.stack(ls).to(device)#[1, 3, 224, 224]
            # print(image.shape)
            # print(texte.shape)
            logits_per_image, logits_per_text = clipmodel(image, texte)
            probs_sort = logits_per_image.softmax(dim=-1)
            topp=probs_sort.sort(descending=True)[1][0,:10]#取前10
            x=numpy.stack([std_pic]*10)
            text = ["a photo of a %s, whiter is closer"%classes[i] for i in topp]
            ls = []
            for i in range(0, len(text), 5) :
                __, x_samples, detected_map = diffusionModel(x[i: i+5], text[i: i+5], 0, 0)
                ls.append(x_samples[0, ...]) # 1 * 5 * 512 * 512 * 3
            x_samples = torch.cat(ls, dim=0)
            image = x_samples 
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
            logits_per_image, logits_per_text = clipmodel(image, texte[topp])
            probs = logits_per_image.softmax(dim=-1)
            log_probs = torch.log(probs)
            topers = (log_probs.sort(dim=0)[0])[:, :]
            guess = (topers.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).argmax().item()
            
            ## try1 get diag max
            # guess = torch.diag(probs).argmax().item()
            
            ## try2 diag max + col sum
            # colsum=(torch.sum(probs,dim=0)-torch.diag(probs)).softmax(dim=-1)
            # colsum = (torch.sum(probs,dim=0)-torch.max(probs,dim=0)[0]*(torch.max(probs,dim=0)[1]!=torch.arange(0,len(probs)).to(device))).softmax(dim=-1)
            
            # colsum = torch.sum(probs*(probs<=torch.diag(probs)),dim=0).softmax(dim=-1)
            # guess = topp[(0.4*colsum+0.6*torch.diag(probs)).argmax()].item()
            tot+=1
            acc += (guess == C)
            with open("./flq/log/%s.log"%filename, "a") as f :
                f.write(str(acc / tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
                f.write("\n")
                if (guess != C) : 
                    ttmp=[classes[i] for i in topp]
                    f.write(str(ttmp))#top10的名字
                    f.write("\n")
                    
                    f.write(str((C in topp))+str(probs_sort.sort(descending=True)[0].cpu().numpy().tolist()))
                    f.write("\n")

                    # debug_img = Image.fromarray(image[C, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                    # debug_img.save("./flq/%s_C.png" %file)
                    # debug_img = Image.fromarray(image[guess, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                    # debug_img.save("./flq/%s_G.png" %file)
                    # f.write(str((log_probs.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).cpu().numpy().tolist()))
                    
                    # f.write(str(colsum.cpu().numpy().tolist()))
                    # f.write("\n")
                    # tmp_put=probs.cpu().numpy().tolist()
                    # for i in range(probs.shape[0]):
                    #     f.write(str(tmp_put[i]))
                    #     f.write(str("\n"))
            print(acc / (tot), tot, "ans=", classes[C], "guess=", classes[guess])
            # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        with open("./flq/log/%s+.log"%filename, "a") as f:
            f.write(str(acc/(tot))+" "+str((tot))+" a="+classes[C] )
            f.write("\n")

def run_no_diffusion(device, projectionModel):
    filename='1vN10V32_nodiffu'
    clipmodel, preprocess = clip.load("ViT-B/32", device=device)
    
    path = "../data/ModelNet10/"
    # classes = ["nightstand", "desk","dresser", "bathtub"]
    classes = [ "monitor", "bed", "chair", "toilet", "table", "sofa", "nightstand", "dresser","desk", "bathtub"]
    # path = "../data/ModelNet40/"
    # classes = ["nightstand", "desk","dresser", "bathtub","monitor", "bed", "chair", "toilet", "table", "sofa"]
    # classes = ["nightstand", "dresser", "desk", "bathtub","monitor", "bed", "chair", "toilet", "table","sofa"]
    # classes = ['stairs', 'curtain', 'sink', 'xbox', 'plant', 'car', 'bed', 'chair', 'lamp', 'door', 'wardrobe', 'cone', 'toilet', 'bench', 'bowl', 'desk', 'airplane', 'bottle', 'mantel', 'radio', 'person', 'cup', 'bathtub', 'monitor', 'sofa', 'vase', 'laptop', 'table', 'keyboard', 'piano', 'tent', 'guitar', 'stool', 'bookshelf', 'dresser', 'TV stand', 'range hood', 'nightstand', 'glass box', 'flower pot']
    # random.shuffle(classes)
    text = ["a photo of a %s, whiter is closer" %_ for _ in classes]
    texte = clip.tokenize(["a photo of a %s" %_ for _ in classes]).to(device)
    # num_test = 20
    for C in range(len(classes)) :
        acc, tot, counter = 0, 0, 0
        full_path = path + classes[C] + "/test/"
        files = os.listdir(full_path)
        # skiprate = len(files)//num_test
        for file in files :
            if file[-1] != 'f' : continue
            counter += 1
            # print(file, counter)
            # if tot>=num_test : break
            # if (counter % skiprate != 0) :continue
         ###############test#################
            with open(full_path + file) as f:
                # assert f.readline().strip() == "OFF"
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
            pointList = numpy.array(pointList)
            pic, multi = projectionModel(pointList.copy(), planeList)
            # 3 * 512 * 512  *1
            std_pic = pic[1, ...]
            
            x = numpy.stack([std_pic]*len(classes))
            
            # 10 * 512 * 512 * 1
            # text = ["%s"%_ for _ in classes]
        
            # ls = []
            # for i in range(0, len(text), 5) :
            #     __, x_samples, detected_map = diffusionModel(x[i: i+5], text[i: i+5], 0, 0)
            #     ls.append(x_samples[0, ...]) # 1 * 5 * 512 * 512 * 3
            # x_samples = torch.cat(ls, dim=0)
            x = numpy.concatenate([x, x, x], axis=-1)
            # print(x.shape)
            image = torch.from_numpy(x).to(device)  # 10 * 512 * 512 * 3

           
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
                f.write(str(acc / tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
                f.write("\n")
                # if (guess != C) : 
                #     # debug_img = Image.fromarray(image[C, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                #     # debug_img.save("./flq/%s_C.png" %file)
                #     # debug_img = Image.fromarray(image[guess, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                #     # debug_img.save("./flq/%s_G.png" %file)
                #     # f.write(str((log_probs.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).cpu().numpy().tolist()))
                #     # f.write(str(colsum.cpu().numpy().tolist()))
                #     # f.write("\n")
                #     tmp_put=probs.cpu().numpy().tolist()
                #     for i in range(probs.shape[0]):
                #         f.write(str(tmp_put[i]))
                #         f.write(str("\n"))
                #     # f.write(str(probs.cpu().numpy().tolist()))
                #     f.write("\n")
            print(acc / (tot), tot, "ans=", classes[C], "guess=", classes[guess])
            # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        with open("./flq/log/%s+.log"%filename, "a") as f:
            f.write(str(acc/(tot))+" "+str((tot))+" a="+classes[C] )
            f.write("\n")


def run_multi_view(device, diffusionModel) :
    projectionModel = ProjectionModel.Perspective4view(
        image_resolution = 512,
        rotate = (-0, -35, -135),
        areadensity = 0.0004
    )    

    filename='4vN10V16_plus2'
    clipmodel, preprocess = clip.load("ViT-B-16", device=device,download_root = "../../encoder_model")
    
    path = "../data/ModelNet10/"
    classes = [ "dresser","nightstand","sofa", "monitor", "desk", "bed", "chair", "toilet", "table", "bathtub"]
    # classes = ["nightstand", "monitor", "bed", "chair", "toilet", "table", "sofa", "dresser", "desk", "bathtub"]
    # classes = [ "monitor", "bed","chair", "toilet", "table", "sofa", "nightstand", "dresser", "desk", "bathtub"]
    # path = "../data/ModelNet40/"
    # classes = ['stairs', 'curtain', 'sink', 'xbox', 'plant', 'car', 'bed', 'chair', 'lamp', 'door', 'wardrobe', 'cone', 'toilet', 'bench', 'bowl', 'desk', 'airplane', 'bottle', 'mantel', 'radio', 'person', 'cup', 'bathtub', 'monitor', 'sofa', 'vase', 'laptop', 'table', 'keyboard', 'piano', 'tent', 'guitar', 'stool', 'bookshelf', 'dresser', 'TV stand', 'range hood', 'nightstand', 'glass box', 'flower pot']
    # random.shuffle(classes)
    # text = ["a photo of a %s, whiter is closer" %_ for _ in classes]
    texte = clip.tokenize(["a 3D image of a %s, with rendered background." %_ for _ in classes]).to(device)
    # num_test = 10
    for C in range(len(classes)) :
        acc, tot, counter = 0, 0, 0
        full_path = path + classes[C] + "/test/"
        files = os.listdir(full_path)
        # skiprate = len(files)//num_test
        for file in files :
            if file[-1] != 'f' : continue
            counter += 1
            # print(file, counter)
            # if (counter % skiprate != 0) :continue
        ###############test#################
            with open(full_path + file) as f:
                # assert f.readline().strip() == "OFF"
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
            pic, multi = projectionModel(numpy.array(pointList), planeList)
            std_pic = pic[:, ...]# 4 * 512 * 512  *1
            print("============================")
            print("std_pic:",std_pic.shape)
            x = numpy.concatenate([std_pic]*len(classes), axis=0)# (C * 4) * 512 * 512  *1
            text = list(itertools.chain(*[["a photo of a %s, whiter is closer"%_]*multi for _ in classes]))
            #10 * 512 * 512 * 1
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
            logit_scale = clipmodel.logit_scale.exp()
            mylogits_per_image = logit_scale * image_features @ text_features.t()
            ###deal image_Prop 4 to 1
            mylogits_per_image = torch.nn.functional.max_pool2d(
                mylogits_per_image[None, ...], (multi, 1)
            )[0, ...]
            # #
            # per_image=torch.zeros(len(mylogits_per_image))
            # for i in range(len(mylogits_per_image)):
            #     per_image[i]=mylogits_per_image[i][i//4]
            # idx=per_image.reshape(len(mylogits_per_image)//4,4).argmax(-1)+torch.arange(0,len(mylogits_per_image),4)
            # mylogits_per_image=mylogits_per_image[idx]
            # probs=torch.log(mylogits_per_image).softmax(dim=-1)

            probs = mylogits_per_image.softmax(dim=-1)


            # log_probs = torch.log(probs)
            # topers = (log_probs.sort(dim=0)[0])[:, :]
            # global_info = topers.mean(dim=0)
            # global_info = torch.exp(global_info)
            # # global_info = global_info.softmax(dim=-1)
            # global_info -= global_info.min()
            # global_info /= global_info.max()
            # guess = (global_info*probs.max(dim=0)[0]).argmax().item()
            # print(topers.sum(dim=0).softmax(dim=-1))
            # print(probs.max(dim=0)[0])
            # guess = (probs.max(dim=0)[0]).argmax().item()
            colsum = torch.sum(probs*(probs<=torch.diag(probs)),dim=0).softmax(dim=-1)
            guess = (0.4*colsum+0.6*torch.diag(probs)).argmax().item()
            acc += (guess == C)
            tot += 1

            with open("./flq/log/%s.log"%filename, "a") as f :
                f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
                f.write("\n")
                # if (guess != C) : 
                #     f.write(str(colsum.cpu().numpy().tolist()))
                #     f.write("\n")
                #     tmp_put=probs.cpu().numpy().tolist()
                #     for i in range(probs.shape[0]):
                #         f.write(str(tmp_put[i]))
                #         f.write(str("\n"))
            print(acc / tot, tot, "ans=", classes[C], "guess=", classes[guess])
            # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        with open("./flq/log/%s+.log"%filename, "a") as f:
            f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C])
            f.write("\n")


def run(device, projectionModel, diffusionModel) :
    filename='1vN10V32'
    clipmodel, preprocess = clip.load("ViT-B/32", device=device)
    
    path = "../data/ModelNet10/"
    classes = ["monitor", "bed", "chair", "toilet", "table", "sofa", "nightstand", "dresser", "desk", "bathtub"]
    # path = "../data/ModelNet40/"
    # classes = ["monitor", "bed", "chair", "toilet", "table", "sofa", "nightstand", "dresser", "desk", "bathtub"]
    # classes = ['stairs', 'curtain', 'sink', 'xbox', 'plant', 'car', 'bed', 'chair', 'lamp', 'door', 'wardrobe', 'cone', 'toilet', 'bench', 'bowl', 'desk', 'airplane', 'bottle', 'mantel', 'radio', 'person', 'cup', 'bathtub', 'monitor', 'sofa', 'vase', 'laptop', 'table', 'keyboard', 'piano', 'tent', 'guitar', 'stool', 'bookshelf', 'dresser', 'TV stand', 'range hood', 'nightstand', 'glass box', 'flower pot']
    # random.shuffle(classes)

    # num_test = 20
    for C in range(len(classes)) :
        acc, tot, counter = 0, 0, 0
        full_path = path + classes[C] + "/test/"
        files = os.listdir(full_path)
        # skiprate = len(files)//num_test
        for file in files :
            if file[-1] != 'f' : continue
            counter += 1
            # print(file, counter)
            # if tot>=num_test : break
            # if (counter % skiprate != 0 ) :continue
        ###############test#################
            with open(full_path + file) as f:
                # assert f.readline().strip() == "OFF"
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
            pic, multi = projectionModel(numpy.array(pointList), planeList)
            # 3 * 512 * 512  *1
            std_pic = pic[1, ...]
            
            x = numpy.stack([std_pic]*len(classes))
            #10 * 512 * 512 * 1
            text = ["a photo of a %s, whiter is closer"%_ for _ in classes]
        
            ls = []
            for i in range(0, len(text), 5) :
                __, x_samples, detected_map = diffusionModel(x[i: i+5], text[i: i+5], 0, 0)
                ls.append(x_samples[0, ...]) # 1 * 5 * 512 * 512 * 3
            x_samples = torch.cat(ls, dim=0)
            
            image = x_samples  # 10 * 512 * 512 * 3

            
            # debug_img = Image.fromarray(image[0, ...].squeeze().cpu().numpy().astype(numpy.uint8))
            # debug_img.save("./debug/tmpbed.png")
            # debug_img = Image.fromarray(image[1, ...].squeeze().cpu().numpy().astype(numpy.uint8))
            # debug_img.save("./debug/tmpbath.png")
            # return 
            # : check passed

            text = ["a photo of a %s" %_ for _ in classes]
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
            # print(image.shape)
            # debug_img = Image.fromarray(image[0, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
            # debug_img.save("./debug/tmpbed.png")
            # debug_img = Image.fromarray(image[1, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
            # debug_img.save("./debug/tmpbath.png")
            # return 

            # image = image.transpose(1,2).transpose(2,3)# 10 * 3 * 224 * 224
            
            logits_per_image, logits_per_text = clipmodel(image, texte)
            probs = logits_per_image.softmax(dim=-1)
            # print("===============")
            # for ii in range(40) :
            #     for jj in range(40) :
            #         print(probs[ii, jj].item(), end=" ")
            #     print()

            log_probs = torch.log(probs)
            topers = (log_probs.sort(dim=0)[0])[:, :]
            guess = (topers.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).argmax().item()
            # print(topers.sum(dim=0).softmax(dim=-1))
            # print(probs.max(dim=0)[0])
            # guess = (probs.max(dim=0)[0]).argmax().item()
            tot +=1
            acc += (guess == C)
            with open("./flq/log/%s.log"%filename, "a") as f :
                f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
                f.write("\n")
                # if (guess != C) : 
                #     # debug_img = Image.fromarray(image[C, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                #     # debug_img.save("./flq/%s_C.png" %file)
                #     # debug_img = Image.fromarray(image[guess, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                #     # debug_img.save("./flq/%s_G.png" %file)
                #     f.write(str((log_probs.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).cpu().numpy().tolist()))
                #     f.write("\n")
                #     f.write(str(probs.cpu().numpy().tolist()))
                #     f.write("\n")
            print(acc / tot, tot, "ans=", classes[C], "guess=", classes[guess])
            # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        with open("./flq/log/%s+.log"%filename, "a") as f:
            f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C])
            f.write("\n")


def test(device, projectionModel, diffusionModel) :  
    for id in [141,155,144] :
        print(id)
        ###############test#################
        obj = "curtain"
        # tid = id%100
        
        # with open("../data/ModelNet10/%s/test/night_stand_0216.off" %obj) as f:
        # with open("../data/ModelNet10/%s/train/%s_00%d%d.off" % (obj, obj, tid//10, tid%10)) as f:
        with open("../data/ModelNet40/%s/test/%s_0%d.off" % (obj, obj, id) )as f:
            assert f.readline().strip() == "OFF"
            n, m, _ = map(int, f.readline().strip().split(" "))
            pointList, planeList = [], []
            for i in range(n) :
                x, y, z = map(float, f.readline().strip().split(" "))
                pointList.append([x, y, z])
            for i in range(m) :
                _, a, b, c = map(int, f.readline().strip().split(" "))
                planeList.append([a, b, c])
        pic, multi = projectionModel(numpy.array(pointList), planeList)
        img = Image.fromarray(pic[0, ...].squeeze())
        img.save("./flq/curtain/%d'0project-example.png"%id)
        img = Image.fromarray(pic[1, ...].squeeze())
        img.save("./flq/curtain/%d'1project-example.png"%id)
        img = Image.fromarray(pic[2, ...].squeeze())
        img.save("./flq/curtain/%d'2project-example.png"%id)
        img = Image.fromarray(pic[3, ...].squeeze())
        img.save("./flq/curtain/%d'3project-example.png"%id)
        print("pic.shape:",pic.shape)
        _, x_samples, detected_map = diffusionModel(pic[:,...], ["a photo of a %s, whiter is closer" % obj]*multi, 0, 0)
        for i in range(x_samples.shape[1]) :
            print("stairs")
            img = Image.fromarray(x_samples[0, i, ...].squeeze().cpu().numpy().astype(numpy.uint8))
            img.save("./flq/curtain/%d'%d-x_samples_%s.png"%(id,i,obj))
        
        obj1="range hood"
        _, x_samples, detected_map = diffusionModel(pic[:,...], ["a photo of a %s, whiter is closer" % obj1]*multi, 0, 0)
        for i in range(x_samples.shape[1]) :
            print(obj1)
            img = Image.fromarray(x_samples[0, i, ...].squeeze().cpu().numpy().astype(numpy.uint8))
            img.save("./flq/curtain/%d'%d-x_samples_%s.png"%(id,i,obj1))

        
        ################
        ### clip model
        # clipmodel, preprocess = clip.load("ViT-B/32", device=device)
        # classes = ["sofa","nightstand", "monitor", "bed", "chair", "toilet", "table",  "dresser", "desk", "bathtub"]
        # text = ["%s" %_ for _ in classes]
        # texte = clip.tokenize(text).to(device)
        # image = x_samples[0,1,...]
        # print("image.shape:",image.shape)
        # # tmp = Image.fromarray(
        # #     image.cpu().numpy().astype(numpy.uint8)
        # # )
        # # tmp.save("./tmp.png")
        # image = preprocess(
        #     Image.fromarray(
        #         image.cpu().numpy().astype(numpy.uint8)
        #     )
        # )
        # print("image.shape:",image.shape)
        # image = image.to(device)
        # print("image.shape:",image.shape)
        # logits_per_image, logits_per_text = clipmodel(image.unsqueeze(0), texte)
        # probs = logits_per_image.softmax(dim=-1)
        # # print(probs)
        # tmp_put=probs.cpu().numpy().tolist()
        # with open("./flq/log/ModelNet10-2023.4.22.log", "a") as f :
        #     for i in range(probs.shape[0]):
        #                     f.write(str(tmp_put[i]))
        #                     f.write(str("\n"))
    
    
    # cloud2textFeatureModel = Cloud2textFeatureModel.BaseLine()
    
    # projectionModel = ProjectionModel.BaseLine(
    #     image_resolution = 512
    # )
    
    # diffusionModel = DiffusionModel.BaseLine(
    #     image_resolution = 512,
    #     n_gene = 1,
    #     device = "cuda:0"
    # )
    
    
    # model = MainModel(cloud2textFeatureModel, 
    #                   projectionModel, 
    #                   diffusionModel, 
    #                   clipModel, 
    #                   classificationHead = CosineSim, 
    #                   multiProjection = False
    #                 )
    
def train(loadPath = None):
    import random
    from tqdm import tqdm
    
    clipmodel, preprocess = clip.load("ViT-B/32", device=device)
    projectionModel = ProjectionModel.Perspective4view_plus_autodensity(
        image_resolution = 512,
        rotate = (-0, -35, -135),
        npoints = 2048
    )    
    dataloader_train, dataloader_test, classes = utils.getDataloader(
        projectionModel,
        path = "../data/ModelNet10/"
    )
    # ##test
    #     for pic, multi, sample_points, y in dataloder_train :
    #         print(pic.shape, multi, sample_points.shape, y, sep="\n")
    # ##test
    diffusionModel = DiffusionModel.BaseLine(
        image_resolution = 512,
        n_gene = 1,
        device = device,
        low_threshold = 5,
        high_threshold = 10
    )
    cloud2textFeatureModel = TransformerModel.TransformerModel(
        npoints = 2048,
        nclass = len(classes),
        fromPretrainPath = "../pretrain_transf/pointtransformer.pt"
    )

    model = MainModel(cloud2textFeatureModel, diffusionModel, clipmodel, preprocess, utils.classificationFn, classes)
    if loadPath is not None :
        model = torch.load("MainModel.pt")

    lossfn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.linear_textfeature.parameters(), lr=1e-3)
    # print(opt.param_groups[]) 

    epochs = 5
    train_log, eval_log = [], []
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
                # print((model.linear_textfeature.weight**2).sum())
                # print(model.linear_textfeature.weight)
                # print(model.linear_textfeature.weight.grad)
                print("grad", (model.linear_textfeature.weight.grad**2).sum())
                # break
            iter.close()
            return (acc/tot, tloss/tot)

        model.train()
        train_log.append(run(dataloader_train, "train"))
        with torch.no_grad() :
            model.eval()
            eval_log.append(run(dataloader_test, "eval"))
            if eval_log[-1][0] > best :
                best = eval_log[-1][0]
                torch.save(model, "MainModel.pt")


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

    seed_everything(1136514422)

##
    # train()
    # sys.exit()
##

    # projectionModel = ProjectionModel.Perspective4view(
    #     image_resolution = 512,
    #     rotate = (-0, -35, -135),
    #     areadensity = 0.0003,
    # )    
    projectionModel = ProjectionModel.Perspective(
        image_resolution = 512,
        rotate = (-0, -35, -135),
        areadensity = 0.0003,
    )    
    diffusionModel = DiffusionModel.BaseLine(
        image_resolution = 512,
        n_gene = 1,
        device = device,
        low_threshold = 5,
        high_threshold = 10,
        bg_threshold = 0.4,
        exttype = "depth",
        detect_resolution=512
        
    )
    # with torch.no_grad():
    #     test(device, projectionModel, diffusionModel)
    # with torch.no_grad():
    #     run(device, projectionModel, diffusionModel)
    # with torch.no_grad():
    #     run_multi_view(device, diffusionModel)
    with torch.no_grad():
        run_flq(device, projectionModel, diffusionModel)
    # with torch.no_grad():
    #     run_no_diffusion(device, projectionModel)
    # with torch.no_grad():
    #     run_N40(device,projectionModel,diffusionModel)