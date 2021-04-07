import json
import os
from PIL import ImageDraw,Image

# parent_path = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/hologram'
json_file = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/annotations/annotations.json'
with open(json_file) as annos:
    annotations = json.load(annos)
annotations = annotations['annotations']
# image_path = os.path.join(parent_path, 'Hologram' + str(0) + '.png')
# image_path = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/test0.png'
image_path = '/data_holo/hologram/0.jpg'
image = Image.open(image_path)
draw = ImageDraw.Draw(image)
for i in range(len(annotations)):
    annotation = annotations[i]
    # if annotation['category_id'] != 1: # 1表示人这一类
    #     continue
    image_id = annotation['image_id']
    if image_id != 0:
        continue
    bbox = annotation['bbox'] # (x1, y1, w, h)
    x, y, w, h = bbox
    # 参数为(图像，左上角坐标，右下角坐标，边框线条颜色，线条宽度)
    # 注意这里坐标必须为整数，还有一点要注意的是opencv读出的图片通道为BGR，所以选择颜色的时候也要注意
    draw.rectangle([int(x), int(y), int(x + w), int(y + h)], width=2, outline='yellow')
    # 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
image.show()