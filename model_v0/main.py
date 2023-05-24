
def run_multi_view(device, projectionModel, diffusionModel) :

    clipmodel, preprocess = clip.load("ViT-B/32", device=device)
    
    path = "../data/ModelNet10/"
    classes = ["nightstand", "monitor", "bed", "chair", "toilet", "table", "sofa", "dresser", "desk", "bathtub"]
    # classes = ["monitor", "bed", "chair", "toilet", "table", "sofa", "nightstand", "dresser", "desk", "bathtub"]
    # path = "../data/ModelNet40/"
    # classes = ['stairs', 'curtain', 'sink', 'xbox', 'plant', 'car', 'bed', 'chair', 'lamp', 'door', 'wardrobe', 'cone', 'toilet', 'bench', 'bowl', 'desk', 'airplane', 'bottle', 'mantel', 'radio', 'person', 'cup', 'bathtub', 'monitor', 'sofa', 'vase', 'laptop', 'table', 'keyboard', 'piano', 'tent', 'guitar', 'stool', 'bookshelf', 'dresser', 'TV stand', 'range hood', 'nightstand', 'glass box', 'flower pot']
    # random.shuffle(classes)

    num_test = 10
    for C in range(len(classes)) :
        acc, tot, counter = 0, 0, 0
        full_path = path + classes[C] + "/test/"
        files = os.listdir(full_path)
        skiprate = len(files)//num_test
        for file in files :
            if file[-1] != 'f' : continue
            counter += 1
            print(file, counter)
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
            pic, multi = projectionModel(numpy.array(pointList), planeList)
            std_pic = pic[:, ...]# 4 * 512 * 512  *1
            
            x = numpy.concatenate([std_pic]*len(classes), axis=0)# (C * 4) * 512 * 512  *1
            #10 * 512 * 512 * 1
            text = list(itertools.chain(*[["%s"%_]*multi for _ in classes]))
        
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

            text = ["%s" %_ for _ in classes]
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
            # ###deal image_features 4 to 1
            # image_features = torch.nn.functional.avg_pool2d(
            #     image_features[None, ...], (multi, 1)
            # )[0, ...]
            # print(text_features.shape)
            # print(text_features.shape)
            ###
            logit_scale = clipmodel.logit_scale.exp()
            mylogits_per_image = logit_scale * image_features @ text_features.t()
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


            log_probs = torch.log(probs)
            topers = (log_probs.sort(dim=0)[0])[:, :]
            global_info = topers.mean(dim=0)
            global_info = torch.exp(global_info)
            # global_info = global_info.softmax(dim=-1)
            # global_info -= global_info.min()
            # global_info /= global_info.max()
            guess = (global_info*probs.max(dim=0)[0]).argmax().item()
            # print(topers.sum(dim=0).softmax(dim=-1))
            # print(probs.max(dim=0)[0])
            # guess = (probs.max(dim=0)[0]).argmax().item()
            print(guess)
            print(C)
            print(probs)

            acc += (guess == C)
            tot += 1

            with open("./log/ModelNet10-2023.4.12.log", "a") as f :
                f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
                f.write("\n")
                if (guess != C) : 
                    f.write(str((log_probs.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).cpu().numpy().tolist()))
                    f.write("\n")
            print(acc / tot, tot, "ans=", classes[C], "guess=", classes[guess])
            # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        with open("./log/ModelNet10-2023.4.12+.log", "a") as f:
            # f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
            f.write("\n")



def run(device, projectionModel, diffusionModel) :

    clipmodel, preprocess = clip.load("ViT-B/32", device=device)
    
    path = "../data/ModelNet10/"
    classes = ["monitor", "bed", "chair", "toilet", "table", "sofa", "nightstand", "dresser", "desk", "bathtub"]
    # path = "../data/ModelNet40/"
    # classes = ["monitor", "bed", "chair", "toilet", "table", "sofa", "nightstand", "dresser", "desk", "bathtub"]
    # classes = ['stairs', 'curtain', 'sink', 'xbox', 'plant', 'car', 'bed', 'chair', 'lamp', 'door', 'wardrobe', 'cone', 'toilet', 'bench', 'bowl', 'desk', 'airplane', 'bottle', 'mantel', 'radio', 'person', 'cup', 'bathtub', 'monitor', 'sofa', 'vase', 'laptop', 'table', 'keyboard', 'piano', 'tent', 'guitar', 'stool', 'bookshelf', 'dresser', 'TV stand', 'range hood', 'nightstand', 'glass box', 'flower pot']
    # random.shuffle(classes)

    skiprate = 10
    for C in range(len(classes)) :
        acc, tot = 0, 0
        full_path = path + classes[C] + "/test/"
        files = os.listdir(full_path)
        for file in files :
            if file[-1] != 'f' : continue
            tot += 1
            print(file, tot)
            if (tot % skiprate != 0) :continue
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
            text = ["%s"%_ for _ in classes]
        
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

            text = ["%s" %_ for _ in classes]
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
            print(image.shape)
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
            topers = (log_probs.sort(dim=0)[0])[:6, :]
            guess = (topers.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).argmax().item()
            # print(topers.sum(dim=0).softmax(dim=-1))
            # print(probs.max(dim=0)[0])
            # guess = (probs.max(dim=0)[0]).argmax().item()

            acc += (guess == C)
            with open("./log/ModelNet10-2023.4.13.log", "a") as f :
                f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
                f.write("\n")
                if (guess != C) : 
                    f.write(str(probs.cpu().numpy().tolist()))
                    debug_img = Image.fromarray(image[C, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                    debug_img.save("./flq/%s_C.png" %file)
                    debug_img = Image.fromarray(image[guess, ...].transpose(0,1).transpose(1,2).squeeze().cpu().numpy().astype(numpy.uint8))
                    debug_img.save("./flq/%s_G.png" %file)
                    f.write(str((log_probs.sum(dim=0).softmax(dim=-1)*probs.max(dim=0)[0]).cpu().numpy().tolist()))
                    f.write("\n")
            print(acc / tot, tot, "ans=", classes[C], "guess=", classes[guess])
            # sys.stderr.write(str(acc/tot) + " " + str(tot) + " " + str(classes[i]))
        with open("./log/ModelNet10-2023.4.13+.log", "a") as f:
            # f.write(str(acc/tot)+" "+str(tot)+" a="+classes[C]+" g="+classes[guess] +  " f= " + file)
            f.write("\n")






def test(device, projectionModel, diffusionModel) :
    import numpy, itertools
    from PIL import Image  
    from Models.MainModel import MainModel
    from MyFunctional.classificationHead import CosineSim
    from Models import Cloud2textFeatureModel, ProjectionModel, DiffusionModel
    
    for _ in range(16,17) :
        ###############test#################
        obj = "nightstand"
        id=141
        tid = id%100
        
        with open("../data/ModelNet10/%s/test/night_stand_0216.off" %obj) as f:
        # with open("../data/ModelNet10/raw/%s/train/%s_00%d%d.off" % (obj, obj, tid//10, tid%10)) as f:
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
        img.save("./project-example0''%d.png"%id)
        img = Image.fromarray(pic[1, ...].squeeze())
        img.save("./project-example1''%d.png"%id)
        img = Image.fromarray(pic[2, ...].squeeze())
        img.save("./project-example2''%d.png"%id)
        img = Image.fromarray(pic[3, ...].squeeze())
        img.save("./project-example3''%d.png"%id)
        return
        print(pic.shape)
        _, x_samples, detected_map = diffusionModel(pic, ["%s, whiter is higher"%obj]*multi, 0, 0)
        print(x_samples.shape)
        for i in range(x_samples.shape[1]) :
            img = Image.fromarray(x_samples[0, i, ...].squeeze().cpu().numpy().astype(numpy.uint8))
            img.save("./project-example%d%d.png"%(i,id))
            
        for i in range(detected_map.shape[0]) :
            img = Image.fromarray(detected_map[i, ...].squeeze().astype(numpy.uint8))
            img.save("./project-example%d'%d.png"%(i,id))

        
        ################
    
    
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
    
    
    
if __name__ == '__main__':
    import sys, itertools
    sys.path.append("../")
    sys.path.append("../ControlNet/ControlNetmain/")
    import torch
    cudaid = 7
    torch.cuda.set_device(cudaid)
    device = "cuda:%d"%cudaid

    import numpy, os, cv2, sys, random
    from CLIP.clip import clip
    from PIL import Image  
    from Models.MainModel import MainModel
    from MyFunctional.classificationHead import CosineSim
    from Models import Cloud2textFeatureModel, ProjectionModel, DiffusionModel

    projectionModel = ProjectionModel.Perspective4view(
        image_resolution = 512,
        rotate = (-0, -35, -135)
    )    
    diffusionModel = DiffusionModel.BaseLine(
        image_resolution = 512,
        n_gene = 1,
        device = device,
        low_threshold = 5,
        high_threshold = 10
    )
    # with torch.no_grad():
    #     test(device, projectionModel, diffusionModel)
    # with torch.no_grad():
    #     run(device, projectionModel, diffusionModel)
    with torch.no_grad():
        run_multi_view(device, projectionModel, diffusionModel)
