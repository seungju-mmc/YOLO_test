from PIL import ImageDraw

def img_show(data):
    image, label = data['image'], data['target']
    image_draw = ImageDraw.Draw(image)
    boxes = label['boxes']
    for box in boxes:
        image_draw.rectangle([(box[0],box[1]), (box[2],box[3])], width=1)
    
    image.show()