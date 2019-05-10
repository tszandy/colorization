#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, n_channels):
        super(Discriminator, self).__init__()

        self.c1 = nn.Conv2d(n_channels, 64, 4, 2, 1)
        self.c2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.c3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.c4 = nn.Conv2d(256, 512, 4, 2, 1)

        self.r1 = nn.LeakyReLU(0.2, inplace=True)
        self.r2 = nn.LeakyReLU(0.2, inplace=True)
        self.r3 = nn.LeakyReLU(0.2, inplace=True)
        self.r4 = nn.LeakyReLU(0.2, inplace=True)

        self.bnc1 = nn.BatchNorm2d(64)
        self.bnc2 = nn.BatchNorm2d(128)
        self.bnc3 = nn.BatchNorm2d(256)
        self.bnc4 = nn.BatchNorm2d(512)
        self.linear1 = nn.Linear(512*16*16,1)
        # l = nn.Linear(3*3*256, 2)'

    def forward(self, x):
        e1 = self.r1(self.bnc1(self.c1(x)))
        e2 = self.r2(self.bnc2(self.c2(e1)))
        e3 = self.r3(self.bnc3(self.c3(e2)))
        e4 = self.r4(self.bnc4(self.c4(e3)))
        e5=e4.view(e4.size()[0],-1)
        e6=self.linear1(e5)
        e7=e6.view(-1)
        return e7
