from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import Tensor
import torchvision
import torch
import numpy as np

from unet import UNet, UNET, Discriminator
import os
import cv2
from dataLoader7 import traindata


dataset=traindata('train_image')
train_loader = DataLoader(dataset=dataset,batch_size=10,shuffle=True)


generator_model = UNET(n_channels=1, n_classes=3)
discriminator_model = Discriminator(n_channels=3)
generator_model.cuda()
discriminator_model.cuda()

criterion = torch.nn.BCEWithLogitsLoss()
generator_optimizer = torch.optim.Adam(generator_model.parameters(), lr=2e-4,betas=(0.5,0.999))
discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=2e-4,betas=(0.5,0.999))
if os.path.exists('unet5.pth'):
    generator_model.load_state_dict(torch.load('./unet5.pth'))

if os.path.exists('discriminator_model.pth'):
    discriminator_model.load_state_dict(torch.load('./discriminator_model.pth'))

number_run_generator=5
number_batchs_per_iteration=20
for t in range(5000):
    loader = enumerate(train_loader)

    for _ in range(number_batchs_per_iteration):
        for _ in range(number_run_generator):

            _, (x_data, y_data, x_target ,y_target) = next(loader)
            x_data = x_data.type(torch.FloatTensor) + Tensor(np.random.normal(size=x_data.size(),loc=0,scale=10))
            x_data, y_data, x_target ,y_target = x_data.type(torch.FloatTensor), y_data.type(torch.FloatTensor), x_target.type(torch.FloatTensor) ,y_target.type(torch.FloatTensor)
            x_data, y_data, x_target ,y_target = x_data.cuda(), y_data.cuda(), x_target.cuda() ,y_target.cuda()
            x_data, y_data, x_target ,y_target = Variable(x_data), Variable(y_data), Variable(x_target) ,Variable(y_target)

            sample = generator_model(x_data)
            fake_result = discriminator_model(sample)
            real_result = discriminator_model(y_data)
            d_loss = criterion(real_result,y_target)+criterion(fake_result,x_target)
            print(t,'d_loss',d_loss.data[0])
            if not d_loss.data.cpu().numpy() < 0.01:
                discriminator_optimizer.zero_grad()
                d_loss.backward()
                discriminator_optimizer.step()
            else:
                del sample,fake_result,real_result,d_loss
        _, (x_data, y_data, x_target ,y_target) = next(loader)
        x_data, y_data, x_target ,y_target = x_data.type(torch.FloatTensor), y_data.type(torch.FloatTensor), x_target.type(torch.FloatTensor) ,y_target.type(torch.FloatTensor)
        x_data, y_data, x_target ,y_target = x_data.cuda(), y_data.cuda(), x_target.cuda() ,y_target.cuda()
        x_data, y_data, x_target ,y_target = Variable(x_data), Variable(y_data), Variable(x_target) ,Variable(y_target)

        sample = generator_model(x_data)
        fake_result = discriminator_model(sample)
        g_loss = criterion(fake_result,y_target)
        print(t,'g_loss',g_loss.data[0])
        if not g_loss.data.cpu().numpy() < 0.01:
            generator_optimizer.zero_grad()
            g_loss.backward()
            generator_optimizer.step()
        else:
            del sample,fake_result,g_loss

    url='./output/661726.png'
    picture=cv2.imread(url,cv2.IMREAD_GRAYSCALE)
    picture=cv2.resize(picture,(256,256))

    data=Tensor(picture.reshape(1,1,256,256))
    data=Variable(data.cuda())

    target_pred=generator_model(data)
    target_pred=target_pred.data.cpu().numpy().reshape((3,256,256))
    target_pred=target_pred.transpose((1,2,0))
    cv2.imwrite('./output/picture_color'+str(t)+'.png',target_pred)


    if t%50==0:
        torch.save(generator_model.state_dict(), './model/unet_7_'+str(t)+'.pth')
        torch.save(discriminator_model.state_dict(), './model/discriminator'+str(t)+'.pth')

    torch.save(generator_model.state_dict(), './unet7.pth')
    torch.save(discriminator_model.state_dict(), './discriminator_model.pth')
