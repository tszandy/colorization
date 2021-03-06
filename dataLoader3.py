from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from utils import *

class traindata(Dataset):
    def __init__(self,url):
        image_grey=[]
        image_UV=[]
        for eachdir in os.listdir(url):
            for pic in os.listdir(os.path.join(url,eachdir)):
                image_grey.append(cv2.imread(os.path.join(os.path.join(url,eachdir),pic),cv2.IMREAD_GRAYSCALE).reshape(1,256,256))
                image=cv2.imread(os.path.join(os.path.join(url,eachdir),pic)).astype('float32')
                image=RBG_to_YUV(image)[:,:,1:3].transpose(2,0,1)
                image_UV.append(image)
        self.x_data=np.array(image_grey)
        self.y_data=np.array(image_UV)
        self.len=len(image_grey)

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len
