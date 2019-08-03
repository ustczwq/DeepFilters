import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.autograd import Variable
from torchvision import models, transforms
from torchsummary import summary

class FilterVisualizer(object):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, model, size=56, upSteps=12, upFactor=1.2):
        self.size = size
        self.upSteps = upSteps
        self.upFactor = upFactor
        self.model = model.to(self.device)
        self.model.eval()

    def visualize(self, channel, lr=0.1, epochs=20):

        sz = self.size
        img = np.random.uniform(0.45 - 0.23, 0.45 + 0.23, (sz, sz, 3))
        
        for i in range(self.upSteps):
            img = self.imgNormalize(img, self.mean, self.std)
            ten = self.img2tensor(img, self.device)
            optimizer = torch.optim.Adam([ten], lr=lr, weight_decay=1e-4)

            for ep in range(epochs):
                optimizer.zero_grad()
                
                output = self.model(ten)
                loss = -nn.functional.softmax(output[0])[channel] * output[0, channel]
                # loss = -nn.functional.softmax(output[0])[channel]
              
                if ep % 5 == 0:
                    print('step:', i, 'size:', sz, 'epoch:', ep, 'loss:', loss.item())
                loss.backward()
                optimizer.step()

            img = self.tensor2img(ten)
            img = self.imgRestore(img, self.mean, self.std)

            self.img = img
            self.imgSave(channel)

            sz = int(sz * self.upFactor)
            img = self.imgUpsample(img, sz)       

    def imgUpsample(self, img, size):
        arr = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)      
        arr = cv2.blur(arr, (3, 3))
        return arr
    
    @staticmethod
    def imgNormalize(img, mean, std):
        arr = img
        h, w, _ = np.shape(arr)
        for x in range(h):
            for y in range(w):
                arr[x, y] -= mean
                arr[x, y] /= std
        return arr
        
    @staticmethod
    def imgRestore(img, mean, std):
        arr = img
        h, w, _ = np.shape(arr)
        for x in range(h):
            for y in range(w):
                arr[x, y] *= std
                arr[x, y] += mean

        return np.clip(arr, 0, 1)

    @staticmethod
    def img2tensor(img, device):
        arr = np.transpose(img, (2, 0, 1))
        ten = torch.tensor(arr, dtype=torch.float32, device=device)
        ten = ten.unsqueeze(0)
        return ten.clone().detach().requires_grad_(True)

    @staticmethod
    def tensor2img(ten):
        arr = ten.data.cpu().numpy()[0]
        arr = np.transpose(arr, (1, 2, 0))
        return arr

    def imgSave(self, channel):
        folder = './outputs/softmax*/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        img = self.img
        plt.imsave(folder + str(channel) + '.jpg', img)

    def showLayers(self):
        summary(self.model, (3, 224, 224))
        
channel = 295

# model = models.vgg16()
# model.classifier._modules['6'] = nn.Linear(4096, 2)
# model.load_state_dict(torch.load('model70.ckpt'))

model = models.vgg16(pretrained=True)

FV = FilterVisualizer(model=model, size=56, upSteps=10, upFactor=1.2)
FV.showLayers()
for c in range(10):
    FV.visualize(channel + c, lr=0.05, epochs=40)
