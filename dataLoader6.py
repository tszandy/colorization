from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from utils import *

class traindata(Dataset):
    def __init__(self,url):
        image_grey=[]
        image_RGB=[]
        for eachdir in os.listdir(url):
            for pic in os.listdir(os.path.join(url,eachdir)):
                line=cv2.imread(os.path.join(os.path.join(url,eachdir),pic),cv2.IMREAD_GRAYSCALE)
                line=RBG_to_dilation(line).reshape(1,256,256)
                image_grey.append(line)
                image=cv2.imread(os.path.join(os.path.join(url,eachdir),pic),cv2.IMREAD_GRAYSCALE).astype('float32').reshape((1,256,256))
                image_RGB.append(image)
        self.x_data=np.array(image_grey)
        self.y_data=np.array(image_RGB)
        self.len=len(image_grey)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len
