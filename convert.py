import cv2
import os
from utils import *
src='input_image'
destine1='resized_image'
destine2='grey_image'
destine3='line_image'
for pic in os.listdir(src):
    img=cv2.imread(os.path.join(src,pic))[:,:,:3]
    img=cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join(destine1,pic),img)

for pic in os.listdir(destine1):
    img=cv2.imread(os.path.join(destine1,pic),cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(destine2,pic),img)
    cv2.imwrite(os.path.join(destine3,pic),RBG_to_dilation(img))
