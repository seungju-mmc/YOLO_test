from PIL import ImageDraw
import torch.nn as nn
import torch


def img_show(data):
    image, label = data['image'], data['target']
    image_draw = ImageDraw.Draw(image)
    boxes = label['boxes']
    for box in boxes:
        image_draw.rectangle([(box[0], box[1]), (box[2], box[3])], width=1)
    image.show()


def get_optimizer(parm, network):
    if parm['name'] == 'adam':
        optimizer = torch.optim.Adam(
            network.parameters(), lr=parm['learning_rate'],
            weight_decay=parm['weight_decay'], eps=1e-7)

    elif parm['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            network.parameters(), lr=parm['learning_rate'], 
            weight_decay=parm['weight_decay'],
            momentum=parm['momentum'])
            
    else:
        raise RuntimeError("optimzier error")

    return optimizer


def parallel_model(module, inp, device_ids, output_device=None):
    if not device_ids:
        return module(inp)
    
    if output_device is None:
        output_device = device_ids[0]
    
    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(inp, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


def calculate_global_norm(network):
    total_norm = 0
    for p in network.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item()**2
    return total_norm
