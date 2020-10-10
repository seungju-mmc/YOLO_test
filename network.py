import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Dataset import ImageNetDataset
from utils import get_optimizer, parallel_model
from torch.utils.data import DataLoader


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, int(hs * ws), int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1, eps=1e-5, momentum=0.1, 
               negative_slope=0.1, is_linear=False):
    if is_linear:
        temp = nn.Sequential(nn.Conv2d(in_num, out_num, kernel_size=kernel_size, 
                             padding=padding, stride=stride, bias=False))
    else:
        temp = nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=kernel_size, 
                      padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_num, eps=eps, momentum=momentum),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
    return temp


class ResidualConv(nn.Module):
    
    def __init__(self, in_num):
        super(ResidualConv, self).__init__()

        mid_num = int(in_num / 2)

        self.layer1 = conv_batch(in_num, mid_num, kernel_size=1, padding=0)
        self.layer2 = conv_batch(mid_num, in_num)

    def forward(self, x):
        
        residual = x
        
        out = self.layer1(x)
        z = self.layer2(out)

        z += residual

        return z


class Darknet53(nn.Module):
    
    def __init__(self, img_size, class_num):
        super(Darknet53, self).__init__()
        self.isize = img_size
        self.class_num = class_num
        self.build_network()

    def build_network(self):
        self.conv001 = conv_batch(3, 32)
        self.conv002 = conv_batch(32, 64, stride=2)

        self.block001 = self.make_layer(64, 1)
        self.conv003 = conv_batch(64, 128, stride=2)

        self.block002 = self.make_layer(128, 2)
        self.conv004 = conv_batch(128, 256, stride=2)

        self.block003 = self.make_layer(256, 8)
        self.conv005 = conv_batch(256, 512,  stride=2)

        self.block004 = self.make_layer(512, 8)
        self.conv006 = conv_batch(512, 1024, stride=2)

        self.block005 = self.make_layer(1024, 4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten() 
        self.fc = nn.Linear(1024, self.class_num)

    def make_layer(self, in_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(ResidualConv(in_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv001(x)
        out = self.conv002(out)
        out = self.block001(out)
        out = self.conv003(out)
        out = self.block002(out)
        out = self.conv004(out)
        out = self.block003(out)
        out = self.conv005(out)
        out = self.block004(out)
        out = self.conv006(out)
        out = self.avgpool(out)  # b, 1024, t,t
        out = out.view(-1, 1024)
        out = self.fc(out)
#         out = torch.softmax(out, dim=1)

        return out


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Darknet53_train:

    def __init__(self, batch_size=128, epoch=10, lr=1e-1, weight_decay=5e-4, 
                 momentum=0.9, device="cpu", division=1, 
                 burn_in=True, load_path=None, parallel=False):

        self.epoch = epoch
        self.batch_size = batch_size
        self.mini_batch = int(self.batch_size/division)
        self.device = torch.device(device)
        self.division = division
        self.dataset = DataLoader(ImageNetDataset(), batch_size=self.mini_batch, shuffle=True)
        self.val_dataset = DataLoader(ImageNetDataset(val_mode=True), batch_size=1, shuffle=True)
        self.network = Darknet53().to(self.device)
        self.parallel_mode = parallel
        
        if (load_path is not None):
            self.network.load_state_dict(torch.load(load_path, map_location=self.device))

        self.epoch = epoch
        parm = {}
        parm['name'] = 'sgd'
        parm['learning_rate'] = lr
        parm['weight_decay'] = weight_decay
        parm['momentum'] = momentum

        self.optimizer = get_optimizer(parm, self.network)
        self.lr = lr
        self.burn_in = burn_in
        self.critieron = nn.CrossEntropyLoss()

    def lr_scheduling(self, step):
        if step < 1001 and self.burn_in:
            lr = self.lr * (step/1004)**4
            for g in self.optimizer.param_groups:
                g['lr'] = lr
    
    def run(self):
        step = 1
        print_interval = 100
        for i in range(self.epoch):
            Loss = []
            Prec = []
            Val_Loss = []
            t_Prec = []
            n = 0
            for data in self.dataset:
                self.network.train()
                image, label = data[0].to(self.device), data[1].to(self.device)
                if self.parallel_mode:
                    hypo = parallel_model(
                        self.network, image,
                        output_device=0, device_ids=[0, 1, 2, 3]
                    )
                
                hypo = self.network.forward(image)
                loss = self.critieron(hypo, label)
                loss.backward()
                n += 1
                if n == self.division:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduling(step)
                    n = 0
                    step += 1
                with torch.no_grad():
                    Loss.append(loss.detach().cpu().numpy())
                    total_num = len(label)
                    idx = torch.argmax(hypo, dim=1)
                    total_true = (idx == label).float().sum()
                    t_precision = total_true/total_num
                    t_Prec.append(t_precision.cpu().detach().numpy())
                
                if step > 1000:
                    print_interval = 100
                
                if step % print_interval == 0 and n == 0:
                    with torch.no_grad():
                        self.network.eval()
                        k = 0
                        for val_data in self.val_dataset:
                            val_image, val_label =\
                                val_data[0].to(self.device), val_data[1].to(self.device)
                            val_hypo = self.network.forward(val_image)
                            val_loss = self.critieron(val_hypo, val_label)
                            idx = torch.argmax(val_hypo, dim=1)
                            total_num = len(val_label)
                            total_true = (idx == val_label).float().sum()
                            precision = total_true/total_num
                            Prec.append(precision.cpu().numpy())
                            Val_Loss.append(val_loss.cpu().numpy())
                            k += 1 
                            if k == 10:
                                break
                            
                    loss = np.array(Loss).mean()
                    val_loss = np.array(Val_Loss).mean()
                    prec = np.array(Prec).mean()
                    tprec = np.array(t_Prec).mean()
                    print("""Epoch: {} // Step: {} // Loss : {:.2f} // Val_Loss : {:.2f} //
                           Prec : {:.3f} // Val_Prec : {:.3f}""".format(
                               (i+1), step, loss, val_loss, tprec, prec))
                    Loss = []
                    Prec = []
                    Val_Loss = []
                    t_Prec = []
            save_path = './dataset/Darknet19.pth'
            torch.save(self.network.state_dict(), save_path)


class Yolov2(nn.Module):
    def __init__(self, aSize=5, catNum=20, device="cpu"):
        super(Yolov2, self).__init__()
        self.aSize = aSize
        self.catNum = catNum
        self.device = torch.device(device)
        self.buildNetwork()
    
    def buildNetwork(self):
        feature = Dakrnet19()
        feature.load_state_dict(torch.load('./dataset/Darknet19.pth', map_location=self.device))
        j = 0
        x = []
        for i in feature.children():
            if j == 0:
                maxpool = i
            else:
                i_list = []
                for ii in i.children():
                    i_list.append(ii)
                i = nn.Sequential(*list(i_list))
                i.training = False
                x.append(i)
            j += 1
        k = [1, 3, 7, 11, 17]
        for j in k:
            x.insert(j, maxpool)

        self.feature1 = nn.Sequential(*list(x)[:13]).to(self.device)
        self.feature2 = nn.Sequential(*list(x)[13:-3]).to(self.device)
        self.feature1.training = False
        self.feature2.training = False

        self.conv1 = conv_batch(1024, 1024)
        self.conv2 = conv_batch(1024, 1024)
        self.conv3 = conv_batch(512, 64, kernel_size=1, padding=0)
        self.reorg = Reorg()
        self.conv4 = conv_batch(1024+64*4, 1024)
        self.output = conv_batch(
            1024, self.aSize*(5+self.catNum), is_linear=True, 
            kernel_size=1, padding=0)

    def train(self):
        # self.feature1.train()
        # self.feature2.train()
        self.conv1.train()
        self.conv2.train()
        self.conv3.train()
        self.conv4.train()
        self.output.train()

    def forward(self, x):
        
        z = self.feature1(x)
        y = self.feature2(z)
        y = self.conv1(y)
        y = self.conv2(y)
        z = self.conv3(z)
        z = self.reorg(z)
        y = torch.cat((z, y), dim=1)
        y = self.conv4(y)
        output = self.output(y)

        return output

   
if __name__ == "__main__":

    darknet19 = Darknet53_train(batch_size=128, device="cuda:2", burn_in=True, division=2)
    darknet19.run()
    # test = Yolov2()
