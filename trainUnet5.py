from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import Tensor
import torchvision
import torch

from unet import UNet,UNET
import os
import cv2
from dataLoader5 import traindata


dataset=traindata('train_image')
train_loader = DataLoader(dataset=dataset,batch_size=10,shuffle=True)


model = UNET(n_channels=1,n_classes=3)
model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

if os.path.exists('unet5.pth'):
    model.load_state_dict(torch.load('./unet5.pth'))

for t in range(5000):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.type(torch.FloatTensor),target.type(torch.FloatTensor)
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        target_pred=model(data)
        loss = criterion(target_pred,target)
        print(t,loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if t%50==0:
        torch.save(model.state_dict(), './model/unet_5_'+str(t)+'.pth')
    torch.save(model.state_dict(), './unet5.pth')
