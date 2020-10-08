import torch
import numpy as np
import datetime

from torch.utils.tensorboard import SummaryWriter
from network import Yolov2
from Dataset import VOCDataset
from utils import get_optimizer, parallel_model
from loss import calculate_loss


class Yolov2Trainer:

    def __init__(self, epoch=160, batch_size=64, division=1, 
                 lr=1e-3, momentum=.9, write_mode=False,
                 weight_decay=5e-4, burn_in=True, load_path=None, 
                 eval_mode=False, device="cpu", parallel_mode=False):
        self.epoch = epoch
        self.mini_batch = int(batch_size/division)
        self.division = division
        self.network = Yolov2(device=device)
        self.device = torch.device(device)
        self.network = self.network.to(self.device)
        self.write_mode = write_mode
        self.parallel_mode = parallel_mode

        if load_path is not None:
            self.network.load_state_dict(torch.load(load_path, map_location=self.device))
        self.network.train()

        parm = {}
        parm['name'] = 'sgd'
        parm['learning_rate'] = lr
        self.lr = lr
        parm['weight_decay'] = weight_decay
        parm['momentum'] = momentum
        self.optimizer = get_optimizer(parm, self.network)

        if eval_mode:
            self.val_dataset = VOCDataset(train_mode=False)
        else:
            self.dataset = VOCDataset()
            self.val_dataset = VOCDataset(train_mode=False)
        self.eval_mode = eval_mode
        self.burn_in = burn_in
        date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        if self.write_mode:
            self.writer = SummaryWriter('./dataset/tensorboard/'+date_time+'/')

    def lr_scheduling(self, step):
        if(step < 1000) and (self.burn_in):
            lr = self.lr * (step/1000)**4
            for g in self.optimizer.param_groups:
                g['lr'] = lr
    
    def tensorboard(self, datas, step):
        loss, xy, wh, conf, cat = datas
        self.writer.add_scalar("Total_loss", loss, step)
        self.writer.add_scalar("xy_loss", xy, step)
        self.writer.add_scalar("wh_loss", wh, step)
        self.writer.add_scalar("conf_loss", conf, step)
        self.writer.add_scalar("cat", cat, step)
        
    def run(self):
        step = 0
        Loss = []
        xyLoss, whLoss, confLoss, catLoss = [], [], [], []
        step_per_epoch = int(len(self.dataset)/self.mini_batch)
        total_num = np.linspace(0, len(self.dataset)-1, len(self.dataset))
        n = 0
        print_interval = 1
        for epoch in range(self.epoch):
            np.random.shuffle(total_num)
            index = total_num.copy()
            self.optimizer.zero_grad()
            for i in range(step_per_epoch):
                self.network.train()
                ind = index[:self.mini_batch].copy()
                index = index[self.mini_batch:]
                batch_img, batch_label = [], []
                
                for j in ind:
                    data = self.dataset[int(j)]
                    img, label = data['image'].to(self.device), data['target']
                    batch_img.append(img)
                    batch_label.append(label)
                
                batch_img = torch.stack(batch_img, dim=0).to(self.device)
                if self.parallel_mode:
                    y_preds = parallel_model(self.network, batch_img, [3, 0, 1, 2], output_device=3)
                else:
                    y_preds = self.network.forward(batch_img)
                total_loss, xy_loss, wh_loss, confid_loss, cat_loss = \
                    calculate_loss(y_preds, batch_label, self.device)

                Loss.append(total_loss)
                xyLoss.append(xy_loss)
                whLoss.append(wh_loss)
                confLoss.append(confid_loss)
                catLoss.append(cat_loss)

                total_loss = total_loss/self.division
                total_loss.backward()
                n += 1
                if n == self.division:
                    step += 1
                    self.lr_scheduling(step)
                    n = 0
                    # print(calculate_global_norm(self.network.output))
                    # print(calculate_global_norm(self.network.conv3))
                    # print(calculate_global_norm(self.network.conv2))
                    # print(calculate_global_norm(self.network.conv1))
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()  
                
                if step % print_interval == 0 and n == 0:
                    Loss = torch.stack(Loss, dim=0).cpu().detach().numpy().mean()
                    xyLoss = torch.stack(xyLoss, dim=0).cpu().detach().numpy().mean()
                    whLoss = torch.stack(whLoss, dim=0).cpu().detach().numpy().mean()
                    confLoss = torch.stack(confLoss, dim=0).cpu().detach().numpy().mean()
                    catLoss = torch.stack(catLoss, dim=0).cpu().detach().numpy().mean()
                    for g in self.optimizer.param_groups:
                        lr = g['lr']
                    # self.tensorboard((Loss, xyLoss, whLoss, confLoss,catLoss),step)
                    print("Epoch : {:4d} // Step : {:5d} // Loss : {:.3f} // xyLoss : {:.3f} //\
                           whLoss : {:.3f} // confLoss : {:.3f} //catLoss : {:.3f} //\
                           Learning rate : {:.5f}".format(
                               epoch+1, step, Loss, xyLoss, whLoss, confLoss, catLoss, lr))
                    
                    Loss = []
                    xyLoss, whLoss, confLoss, catLoss = [], [], [], []

                if step > 1000:
                    print_interval = 100

                
if __name__ == "__main__":

    trainer = Yolov2Trainer(batch_size=64, device="cuda:2", division=1)
    trainer.run()
