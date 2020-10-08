import torch

import torch.nn as nn
import numpy as np


def calculate_ious(boxes, box, wh=False, xywh=False):
    """
    boxes have the shape (x0, y0, x1,y1) *n
    box has the shape (x0, y0, x1,y1)
    if wh is True,
    boxes have the shape (w,h),
    box has the shape(x0,y0,x1,y1)
    """

    if wh:
        w_ = box[2]-box[0]

        h_ = box[3]-box[1]

        area1 = w_ * h_
        area2 = boxes[:, 0] * boxes[:, 1]

        w_min = torch.min(boxes[:, 0], w_)
        h_min = torch.min(boxes[:, 1], h_)

        intersection = w_min * h_min

        iou = (intersection)/(area1 + area2 - intersection + 1e-3)
    else:
        if xywh:
            tx0 = boxes[:, 0] - boxes[:, 2]/2
            ty0 = boxes[:, 1] - boxes[:, 3]/2
            tx1 = boxes[:, 0] + boxes[:, 2]/2
            ty1 = boxes[:, 1] + boxes[:, 3]/2
        else:
            tx0, ty0, tx1, ty1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        
        area1 = (x1-x0) * (y1-y0)
        area2 = (tx1-tx0) * (ty1-ty0)

        xmax = torch.min(x1, tx1)
        ymax = torch.min(y1, ty1)
        xmin = torch.max(x0, tx0)
        ymin = torch.max(y0, ty0)

        wid = torch.clamp(xmax - xmin, min=0)
        hei = torch.clamp(ymax - ymin, min=0)
        intersection = wid * hei
        iou = intersection/(area1+area2-intersection+1e-3)
    
    iou = torch.clamp(iou, min=0)
    return iou


def calculate_loss(y_preds, labels, device, l_coord=5, l_confid=1, l_noobj=0.5, 
                   threshold=0.6, catNum=20, 
                   anchor_box=np.load('./dataset/anchor_box.npy'), img_size=416):

    grid_size = y_preds.shape[2]
    batch_size = y_preds.shape[0]
    anchor_size = len(anchor_box)
    reduction = grid_size/img_size
    total_index = int(grid_size**2 * anchor_size)

    OFFSET = torch.zeros((grid_size, grid_size, anchor_size, 2)).float().to(device)

    for i in range(grid_size):
        for j in range(grid_size):
            OFFSET[i, j, :, :] += torch.tensor([j, i]).float().to(device)
    
    anchor_box = anchor_box.copy()
    anchor_box *= grid_size
    anchor_box = torch.tensor(anchor_box).to(device).float()
    y_preds = y_preds.permute(0, 2, 3, 1)  # (B,H,W,C)
    y_preds = y_preds.view((batch_size, grid_size, grid_size, anchor_size,  5+catNum))

    predXY = y_preds[:, :, :, :, :2].sigmoid() + OFFSET
    predXY = predXY.permute(4, 0, 1, 2, 3)
    predXY = predXY.view((2, -1))
    predXY = predXY.permute((1, 0))
    
    predWH = torch.exp(y_preds[:, :, :, :, 2:4])
    predWH = predWH * anchor_box
    predWH = predWH.permute(4, 0, 1, 2, 3)
    predWH = predWH.view((2, -1))
    predWH = predWH.permute(1, 0)

    predConfidenc = y_preds[:, :, :, :, 4:5].sigmoid()
    predConfidenc = predConfidenc.permute(4, 0, 1, 2, 3)
    predConfidenc = predConfidenc.view((1, -1))
    predConfidenc = predConfidenc.permute(1, 0)
    predCat = y_preds[:, :, :, :, 5:].contiguous()
    predCat = predCat.permute(4, 0, 1, 2, 3)
    predCat = predCat.view((20, -1))
    predCat = predCat.permute(1, 0)
    crossentropy = nn.CrossEntropyLoss()

    xy_loss, wh_loss, cf_loss, cat_loss = 0, 0, 0, 0
    total_loss = 0
    for i, label in enumerate(labels): 
        boxes = label['boxes'] * reduction  # x0, y0, x1, y1 in grid scale
        cats = label['category']

        batchXY = predXY[i*total_index:(i+1)*total_index, :]
        batchWH = predWH[i*total_index:(i+1)*total_index, :]
        batchBox = torch.cat((batchXY, batchWH), dim=1)
        batchConfid = predConfidenc[i*total_index:(i+1)*total_index, :]
        objmask = torch.zeros_like(batchConfid).view(-1)
        batchCat = predCat[i*total_index:(i+1)*total_index, :]
        for box, cat in zip(boxes, cats):
            box = box.to(device).float()
            cat = cat.to(device)
            x_true, y_true = (box[0] + box[2])/2, (box[1]+box[3])/2 
            x_ind, y_ind = x_true.long(), y_true.long()
            anchorIous = calculate_ious(anchor_box, box, wh=True)
            anchorTrueIndex = torch.argmax(anchorIous)
            index = y_ind * grid_size * anchor_size + x_ind * anchor_size + anchorTrueIndex
            if index > total_index:
                continue
            selectedXY, selectedWH, selectedConfid, selectedCat = \
                batchXY[index], batchWH[index], batchConfid[index], batchCat[index]

            ious = calculate_ious(batchBox, box, xywh=True)
            objDectInd = (ious > threshold).float()
            objmask += objDectInd
            objmask[index] += 1

            xy = torch.stack((x_true, y_true), dim=0)
            xy_loss += (selectedXY-xy).pow(2).sum() * l_coord

            w, h = box[2]-box[0], box[3]-box[1]
            print(w, h)
            wh = torch.stack((w, h), dim=0).pow(0.5)
            selectedWH = (selectedWH+1e-3).pow(0.5)
            wh_loss += (selectedWH-wh).pow(2).sum() * l_coord

            cf_loss += (selectedConfid - ious[index]).pow(2).sum() * l_confid
            cat_loss += crossentropy(selectedCat.view((1, 20)), cat.view(1))
        
        noobjInd = objmask < 1
        noobjConfid = batchConfid[noobjInd]
        cf_loss += noobjConfid.pow(2).sum() * l_noobj
    total_loss = xy_loss + wh_loss + cf_loss+cat_loss

    return (total_loss/batch_size, xy_loss/batch_size, wh_loss/batch_size,
            cf_loss/batch_size, cat_loss/batch_size)
    
        






            






