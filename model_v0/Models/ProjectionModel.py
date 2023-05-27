import torch, numpy
from tqdm import tqdm
from math import cos, sin, pi

# picSize = 224
class BaseLine(torch.nn.Module) :
    def __init__(self, image_resolution, rotate = (0,0,0), areadensity = 0.00001, linedensity = 0.02, multi = 3) :
        super(BaseLine, self).__init__()
        self.image_resolution = image_resolution
        self.areadensity = areadensity
        self.linedensity = linedensity
        self.multi = multi
        
        beta = rotate[0]/180*pi
        Rx = numpy.array([[1,0,0],
                          [0, cos(beta), -sin(beta)],
                          [0, sin(beta), cos(beta)]])
        Rx_ = numpy.array([[1,0,0],
                          [0, cos(-beta), -sin(-beta)],
                          [0, sin(-beta), cos(-beta)]])
        
        beta = rotate[1]/180*pi
        Ry = numpy.array([[cos(beta), 0, sin(beta)],
                          [0, 1, 0], 
                          [-sin(beta), 0, cos(beta)]])
        Ry_ = numpy.array([[cos(-beta), 0, sin(-beta)],
                          [0, 1, 0], 
                          [-sin(-beta), 0, cos(-beta)]])
        beta = rotate[2]/180*pi
        Rz = numpy.array([[cos(beta), -sin(beta), 0],
                          [sin(beta), cos(beta), 0],
                          [0, 0, 1]])
        Rz_ = numpy.array([[cos(-beta), -sin(-beta), 0],
                          [sin(-beta), cos(-beta), 0],
                          [0, 0, 1]])
        self.Rxyz = Rz@Rx@Ry
        
    def analysisParam(self, p3) :
        getlen = lambda v1,v2 : numpy.sqrt(((v1-v2)**2).sum())
        a = getlen(p3[0], p3[1])
        b = getlen(p3[1], p3[2])
        c = getlen(p3[0], p3[2])
        return a,b,c
        
    def clac_area(self, p3) :
        a,b,c = self.analysisParam(p3)
        p = (a+b+c)/2
        return numpy.sqrt(abs(p*(p-a)*(p-b)*(p-c)))
    
    def sprinkarea(self, p3, num) :
        i = p3[1]-p3[0]
        j = p3[2]-p3[0]
        r = numpy.random.rand(num, 2)
        a = (r**2).sum(axis=1, keepdims=True)
        b = ((1-r)**2).sum(axis=1, keepdims=True)
        std_r = r*(a<b) + (1-r)*(a>b)
        return std_r[:, 0:1]*i + std_r[:, 1:2]*j + p3[0] 
        
    def sprinkline(self, p3) :
        a,b,c = self.analysisParam(p3)
        l = a+b+c
        na, nb, nc = list(map(lambda x: int(x/self.linedensity)+1, [a,b,c]))
        ret = numpy.random.rand(na+nb+nc, 1)
        return numpy.concatenate([
            ret[0: na, 0:1]*(p3[1]-p3[0])+p3[0],
            ret[na: na+nb, 0:1]*(p3[2]-p3[1])+p3[1],
            ret[na+nb: na+nb+nc, 0:1]*(p3[0]-p3[2])+p3[2]
        ], axis = 0)        
              
    def draw_on_2d_base(self, x, y, c) :
        c_max = c.max()
        c/=c_max
        c*=255
        
        pic = numpy.zeros([self.image_resolution, self.image_resolution, 1], dtype = numpy.uint8)
                         
        x = self.image_resolution - x
        y = self.image_resolution - y
        x = x.clip(0, self.image_resolution-1).astype(int)
        y = y.clip(0, self.image_resolution-1).astype(int) 

        index = c.argsort()
        x, y, c = y[index], x[index], c[index]
        pic[x, y, 0] = numpy.maximum(pic[x, y, 0], c)
        # for i in tqdm(range(x.shape[0]), desc = "draw2d") :
        #     pic[int(y[i]), int(x[i]), 0] = max(pic[int(y[i]), int(x[i]), 0], c[i])
            
        return 255-pic

    def draw_on_2d(self, x, y, c) :
        return self.draw_on_2d_base(x, y, c) 
    
    def project(self, newList) :
        # pic = numpy.zeros([self.multi, self.image_resolution, self.image_resolution, 1], dtype = numpy.uint8)
        # 1 2 3 -> 2 3 1
        if (self.multi == 3) :  
            newList = self.rotation(newList) 
            Min = newList.min()
            newList -= Min
            scale = newList.max()
            newList /= scale
            newList *= self.image_resolution
                    
            return numpy.stack([
                self.draw_on_2d(newList[:, 0].copy(), newList[:, 1].copy(), newList[:, 2].copy()),
                self.draw_on_2d(newList[:, 1].copy(), newList[:, 2].copy(), newList[:, 0].copy()),
                self.draw_on_2d(newList[:, 0].copy(), newList[:, 2].copy(), newList[:, 1].copy())
            ])
        elif self.multi == "4view" :
            ls = []
            for i in range(4) :
                tmp = newList.copy()
                cur = self.rotation(tmp)
                cur -= cur[:, 1:].min()
                cur /= cur[:, 1:].max()
                cur *= self.image_resolution

                ls.append(
                    self.draw_on_2d(cur[:, 1], cur[:, 2], cur[:, 0])
                )
                newList = newList@self.extRz
            return numpy.stack(ls)
        else: raise NotImplementedError
          
    def rotation(self, newList) :
        return newList@self.Rxyz

    def forward(self, pointList: numpy , planeList) : # unbatch
        pointList -= pointList.min()
        pointList = pointList/pointList.max()*20-10

        ls = [pointList.copy()]
        # for i in tqdm(range(len(planeList)), desc = "project_expend") :
        for i in range(len(planeList)) :
            area = self.clac_area(pointList[planeList[i]])
            num = int(area/self.areadensity)
            ls.append(self.sprinkarea(pointList[planeList[i]], num))
            ls.append(self.sprinkline(pointList[planeList[i]]))
        newList = numpy.concatenate(ls,axis = 0)
        # newList = numpy.concatenate([
        #     newList,
        #     newList+numpy.array([0,0,.04]),
        #     newList+numpy.array([0,.04,0]),
        #     newList+numpy.array([.04,0,0])
        # ])
        
        multi = self.multi
        if self.multi == "4view" : multi = 4
        return self.project(newList), multi
        
        
        # begin projection : (newList)
        # raise NotImplementedError
    
    
    
        # return cloud[:, :, 0:2] # project to x-y
    
class Perspective(BaseLine) :
    def __init__(self, image_resolution, rotate = (0,0,0), areadensity = 0.0002, linedensity = 0.01, multi = 3, obv_dis = 700) :
        super(Perspective, self).__init__(image_resolution, rotate, areadensity, linedensity, multi)
        self.obv_dis = obv_dis
        print(multi)

    def draw_on_2d_persp(self, x,y,c) :
        O = numpy.array([x.mean(), y.mean(), -self.obv_dis])
        plist = numpy.stack([x,y,self.image_resolution-c]).transpose() # n * 3
        plist -= O
        rate = self.obv_dis / plist[:, 2]
        plist *= rate[:, None]
        plist += O #2 D

        plist -= plist[:, 0:2].min()
        plist = plist/(plist[:, 0:2].max())*self.image_resolution
        
        rate -= rate.min()
        r_max = rate.max()
        rate = rate/r_max
        rate*=255
        
        pic = numpy.zeros([self.image_resolution, self.image_resolution, 1], dtype = numpy.uint8)

        plist = (self.image_resolution - plist).astype(int)  
        plist[plist>=self.image_resolution] = self.image_resolution-1
        # plist = plist.clip(0, self.image_resolution-1).astype(int)
        
        index = rate.argsort()
        x, y, c = plist[index, 1], plist[index, 0], rate[index]
        
        def deal(_x, _y) :
            pic[_x, _y, 0] = numpy.maximum(pic[_x, _y, 0], c)  
        deal(x,y)
        deal((x+1).clip(0, self.image_resolution-1), y)
        deal(x, (y+1).clip(0, self.image_resolution-1))
        # deal((x+1).clip(0, self.image_resolution-1), (y+1).clip(0, self.image_resolution-1))
        # pic[(x+1).clip(0, self.image_resolution-1), y, 0] = numpy.maximum(pic[x, y, 0], c)  

        # for i in range(x.shape[0]) :
        #     pic[int(plist[i, 1]), int(plist[i, 0]), 0] = max(pic[int(plist[i, 1]), int(plist[i, 0]), 0], int(rate[i]))
            
        return pic
    
    def draw_on_2d(self, x, y, c) :
        return self.draw_on_2d_persp(x,y,c)


class Perspective4view(Perspective) :
    def __init__(self, image_resolution, rotate = (0,0,0), areadensity = 0.0002, linedensity = 0.01, multi = "4view", obv_dis = 700) :
        super(Perspective4view, self).__init__(image_resolution, rotate, areadensity, linedensity, multi, obv_dis)

        beta = 90.0/180*pi
        self.extRz = numpy.array([[cos(beta), -sin(beta), 0],
                                  [sin(beta), cos(beta), 0],
                                  [0, 0, 1]]
                                  )

class Perspective4view_plus(Perspective4view) :
    def __init__(self, image_resolution, rotate = (0,0,0), areadensity = 0.0005, linedensity = 0.01, multi = "4view", obv_dis = 700, npoints = 2048) :
        super(Perspective4view_plus, self).__init__(image_resolution, rotate, areadensity, linedensity, multi, obv_dis)

        self.npoints = npoints

    def forward(self, pointList: numpy , planeList) : # unbatch
        pointList -= pointList.min()
        pointList = pointList/pointList.max()*20-10

        ls = [pointList.copy()]
        # for i in tqdm(range(len(planeList)), desc = "project_expend", leave = False) :
        for i in range(len(planeList)) :
            area = self.clac_area(pointList[planeList[i]])
            num = int(area/self.areadensity)
            ls.append(self.sprinkarea(pointList[planeList[i]], num))
            ls.append(self.sprinkline(pointList[planeList[i]]))
        newList = numpy.concatenate(ls,axis = 0)
        # newList = numpy.concatenate([
        #     newList,
        #     newList+numpy.array([0,0,.04]),
        #     newList+numpy.array([0,.04,0]),
        #     newList+numpy.array([.04,0,0])
        # ])
        
        multi = self.multi
        if self.multi == "4view" : multi = 4
        sample = newList[numpy.random.choice(newList.shape[0], self.npoints), ...]
        return self.project(newList), multi, sample
    
class Perspective4view_plus_autodensity(Perspective4view_plus) :
    def __init__(self, image_resolution, rotate = (0,0,0), areadensity = 0.0005, linedensity = 0.01, multi = "4view", obv_dis = 700, npoints = 2048, ndensity = (3600000,300000)) :
        super(Perspective4view_plus_autodensity, self).__init__(image_resolution, rotate, areadensity, linedensity, multi, obv_dis, npoints)

        self.ndensity = ndensity

    def forward(self, pointList: numpy , planeList) : # unbatch
        pointList -= pointList.min()
        pointList = pointList/pointList.max()*20-10

        totarea, totlength = 0, 0

        # for i in tqdm(range(len(planeList)), desc = "calc", leave=False) :
        for i in range(len(planeList)) :
            totarea += self.clac_area(pointList[planeList[i]])
            a,b,c = self.analysisParam(pointList[planeList[i]])
            totlength += a+b+c

        self.areadensity = totarea / self.ndensity[0]
        self.linedensity = (totlength/2) / self.ndensity[1]
        # print("density", self.areadensity, self.linedensity)

        ls = [pointList.copy()]
        # for i in tqdm(range(len(planeList)), desc = "project_expend", leave = False) :
        for i in range(len(planeList)) :
            area = self.clac_area(pointList[planeList[i]])
            num = int(area/self.areadensity)
            ls.append(self.sprinkarea(pointList[planeList[i]], num))
            ls.append(self.sprinkline(pointList[planeList[i]]))
        newList = numpy.concatenate(ls,axis = 0)
        print(newList.shape)
        # newList = numpy.concatenate([
        #     newList,
        #     newList+numpy.array([0,0,.04]),
        #     newList+numpy.array([0,.04,0]),
        #     newList+numpy.array([.04,0,0])
        # ])
        
        multi = self.multi
        if self.multi == "4view" : multi = 4
        sample = newList[numpy.random.choice(newList.shape[0], self.npoints), ...]
        return self.project(newList), multi, sample

