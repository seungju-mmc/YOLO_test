import torch
import numpy as np
from network import Yolov2
from Dataset import VOCDataset
from torchvision.ops import nms
from torchvision import transforms
from utils import img_show


class EvalMAP:

    def __init__(self, network, grid_size=13):
        self.network = network
        self.network.eval()
        self.gSize = grid_size
        self.OFFSET = torch.zeros((grid_size, grid_size, 5, 2)).float().cpu()
        for i in range(grid_size):
            for j in range(grid_size):
                self.OFFSET[i, j, :, :] += torch.tensor([j, i]).float().cpu()
        
        self.cat = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']
        self.PIL = transforms.ToPILImage()
    
    def getBBoxes(self, x, aBox='./dataset/anchor_box.npy', threshold=0.2):
        aBox = np.load(aBox)
        pred = self.network.forward(x)
        pred = pred.permute((0, 2, 3, 1))
        pred = pred.view((1, self.gSize, self.gSize, 5, -1))
        pred[:, :, :, :, :2] = pred[:, :, :, :, :2].sigmoid() + self.OFFSET
        pred[:, :, :, :, 2:4] = pred[:, :, :, :, 2:4].exp() * aBox * self.gSize
        pred[:, :, :, :, 5:] = pred[:, :, :, :, 5:].softmax(dim=-1)
        pred[:, :, :, :, 4:5] = pred[:, :, :, :, 4:5].sigmoid() 
        
        index = pred[0, :, :, :, 4] > threshold  # 1, gS, gS
        BBoxes = pred[0, index]

        return BBoxes
    
    def NMS(self, boxes):
        xyXY = torch.zeros_like(boxes[:, :4])
        xyXY[:, 0] = boxes[:, 0] - boxes[:, 2]/2
        xyXY[:, 1] = boxes[:, 1] - boxes[:, 3]/2
        xyXY[:, 2] = boxes[:, 0] + boxes[:, 2]/2
        xyXY[:, 3] = boxes[:, 1] + boxes[:, 3]/2
        scores = boxes[:, 4]
        nmsInd = nms(xyXY, scores, iou_threshold=0.5)
        nmsBoxes = boxes[nmsInd]

        return nmsBoxes
    
    def wrapperBox(self, x, image):
        data = {}
        output = {}
        if x.shape[0] != 0:
            xyXY = torch.zeros_like(x[:, :4])
            xyXY[:, 0] = x[:, 0] - x[:, 2]/2
            xyXY[:, 1] = x[:, 1] - x[:, 3]/2
            xyXY[:, 2] = x[:, 0] + x[:, 2]/2
            xyXY[:, 3] = x[:, 1] + x[:, 3]/2
            output['boxes'] = xyXY * 416 / self.gSize
            cats = torch.argmax(x[:, 5:], dim=1)
            cats_str = []
            for cat in cats:
                cats_str.append(self.cat[cat])
            output['category'] = cats_str
        else:
            output['boxes'] = None
        
        data['image'] = self.PIL(image[0, :, :, :])
        data['target'] = output

        return data

    def forward(self, x, display_mode=True):
        image = x
        bBoxes = self.getBBoxes(x)
        nmsBoxes = self.NMS(bBoxes)
        data = self.wrapperBox(nmsBoxes, image)
        if display_mode:
            img_show(data)

        return data


if __name__ == "__main__":
    network = Yolov2(device="cpu")
    device = torch.device("cpu")
    x = torch.load('./dataset/Yolov2.pth', map_location=device)
    network.load_state_dict(x())
    network.eval()
    Eval = EvalMAP(network)
    
    valDataset = VOCDataset(train_mode=False)

    total_dataset = len(valDataset)
    ind = np.random.randint(0, total_dataset, 20)
    j = 0
    for i in ind:
        data = valDataset[i]
        # img_show(data)
        img, label = data['image'], data['target']
        img = torch.unsqueeze(img, 0)
        j += 1
        with torch.no_grad():
            
            Eval.forward(img)
        

