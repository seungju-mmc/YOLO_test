import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1, eps=1e-5, momentum=0.1, negative_slope=0.1,is_linear=False):
    if is_linear:
        temp = nn.Sequential(
            nn.Conv2d(in_num, out_num,kernel_size=kernel_size, padding=padding,stride=stride,bias=False) 
        )
    else:
        temp = nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
            nn.BatchNorm2d(out_num,eps=eps,momentum=momentum)
            nn.LeakyReLU(negative_slope=negative_slope=)
        )
    return temp

class Dakrnet19(nn.Module):

    def __init__(self):
        super(Dakrnet19, self).__init__()

        self.conv1 = conv_batch(3,32)
        self.maxpool1 = nn.MaxPool2d(32, stride=2)
        self.conv2 = conv_batch(32,64)
        self.maxpool2 = nn.MaxPool2d(64,stride=2)
        self.conv3 = conv_batch(64,128)
        self.conv4 = conv_batch(128,64,kernel_size=1,padding=0)
        self.conv5 = conv_batch(64,128)
        self.maxpool3 = nn.MaxPool2d(128, stride=2)
        self.conv6 = conv_batch(128,256)
        self.conv7 = conv_batch(256,128,kernel_size=1,padding=0)
        self.conv8 = conv_bath(128,256)
        self.maxpool4 = nn.MaxPool2d(256,stride=2)
        self.conv9 = conv_batch(256,512)
        self.conv10 = conv_batch(512,256,kernel_size=1,padding=0)
        self.conv11 = conv_batch(256,512)
        self.conv12 = conv_batch(512,256,kernel_size=1,padding=0)
        self.conv13 = conv_batch(256,512)
        self.maxpool5 = nn.MaxPool2d(512, stride=2)
        self.conv14 = conv_batch(512,1024)
        self.conv15 = conv_batch(1024,512,kernel_size=1,padding=1)
        self.conv16 = conv_batch(512,1024)
        self.conv17 = conv_batch(1024,512, kernel_size=1,padding=0)
        self.conv18 = conv_batch(512,1024)
        self.conv19 = conv_batch(1024, 1000, kernel_size=1, padding=0)
        self.avg_pool = nn.AvgPool2d()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool3(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool4(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool5(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv3(x)