from utils import *
from propagators import ASM
import numpy as np
import pandas as pd
import time
from PIL import Image,ImageDraw
from pycocotools.coco import COCO
import os

def get_xylocation(size,number):
    x = np.random.random(number)*size-float(size/2)
    y = np.random.random(number)*size-float(size/2)
    return x, y
def get_zlocation(z_list,number):
    z = np.random.choice(z_list,number,replace=True)
    return z
def get_size(smin,smax,number):
    s = (smax-smin)*np.random.random(number)+smin
    return s
def get_buffer(z,size):
    z_rate = (z-1*cm)/(3*cm-1*cm)
    # size_rate = (size-size_range[0])/(size_range[1]-size_range[0])
    size_rate = 1
    buffer = 30+30*z_rate+15*size_rate
    # p_size = int(size_range[1]/frame * N)
    # buffer = p_size*10*(z_rate*0.6+size_rate*0.4)
    return buffer
def get_bbox(x,y,z,size):
    px = int(x/frame*N+N/2)
    py = int(N/2+y/frame*N)
    # p_size = int(size/frame*N)
    # buffer = p_size*10
    buffer = get_buffer(z,size)
    bbox_x = max(0, px-buffer)
    bbox_y = max(0, py-buffer)
    height = buffer*2
    width = buffer*2
    if bbox_x+width > N:
        width = N-bbox_x
    if bbox_y+height > N:
        height = N-bbox_y
    seg = [bbox_x,bbox_y,bbox_x,bbox_y+height,bbox_x+width,bbox_y+height,bbox_x+width,bbox_y]
    return (bbox_x,bbox_y,width,height,seg)
def get_new_bbox(old_bbox, boundary):
    # old bbox 里面是以左上为原点
    [o_x1, o_y1, o_x2, o_y2] = old_bbox[0], old_bbox[1],old_bbox[0]+old_bbox[2],old_bbox[1]+old_bbox[3]
    [b_x1, b_y1, b_x2, b_y2] = boundary[0], boundary[2],boundary[1],boundary[3]
    o_cx,o_cy = (o_x2+o_x1)/2,(o_y1+o_y2)/2

    if b_x1<o_cx<b_x2 and b_y1<o_cy<b_y2:
        n_x1 = max(o_x1, b_x1)
        n_y1 = max(o_y1, b_y1)
        n_x2 = min(o_x2, b_x2)
        n_y2 = min(o_y2, b_y2)

        n_w = n_x2-n_x1
        n_h = n_y2-n_y1

        if n_w <=0 or n_h<=0:
            return []
        else:
            # axis center transfer
            n_x1 = n_x1-b_x1
            n_y1 = n_y1-b_y1
            return [n_x1,n_y1,n_w,n_h]
    else:
        return []
def get_new_annos(old_annos,boundary,img_id):
    global ANNO_CNT
    new_annos = []
    for anno in old_annos:
        new_bbox = get_new_bbox(anno['bbox'],boundary)
        if len(new_bbox) != 4:
            continue
        new_annos.append({
            'area': new_bbox[2]*new_bbox[3],
            'bbox': new_bbox,
            'category_id': anno['category_id'],
            'id': ANNO_CNT,
            'image_id': img_id,
            'iscrowd': 0,
        })
        ANNO_CNT += 1
    return new_annos
def particle_field(number,xyrange,z_list,size_range):
    df = pd.DataFrame()
    x,y = get_xylocation(xyrange,number)
    z = get_zlocation(z_list,number)
    s = get_size(size_range[0],size_range[1],number)
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df['size'] = s
    return df
def generate_holo_fromcsv2(file):
    particles_field = pd.read_csv(file)
    z_list = particles_field['z'].unique()
    F_obj = Begin(frame, wavelength, N)
    for i in range(len(z_list)-1):
        prop_dis = z_list[i]-z_list[i+1]
        if prop_dis<0:
            raise ValueError("z is not aceding")
        particles = particles_field[particles_field['z']==z_list[i]]
        for j in range(len(particles)):
            F_obj = CircScreen(F_obj, particles.iloc[j]['size'], particles.iloc[j]['x'], particles.iloc[j]['y'])
        F_obj = ASM(F_obj, prop_dis)

    particles = particles_field[particles_field['z'] == z_list[-1]]
    for j in range(len(particles)):
        F_obj = CircScreen(F_obj, particles.iloc[j]['size'], particles.iloc[j]['x'], particles.iloc[j]['y'])
    F_obj = ASM(F_obj, z_list[-1])
    I = Intensity(F_obj)
    # plt.imsave("Hologram%d.png" % n, I, cmap='gray')
    return I

if __name__ == '__main__':
    wavelength = 633*nm
    N = 1024
    # pixel_pitch = 10*um
    frame = 10*mm # 10mm * 10mm
    size_range = [50*um,50*um]
    depth_range= (1*cm, 3*cm)
    particle_number = (5,6)
    hologram_number = 2
    res_z = (3 * cm - 1 * cm) / 256

    dataset = {}
    dataset['info'] = []
    dataset['licenses'] = []
    dataset['info'].append({'year': "2021", "version": '1',
                            "description": "Hologram synthetic data for test 100ppp",
                            "contributor": "zhangyp",
                            "url": "None",
                            "date_created": "2021-08-05"})
    dataset['licenses'].append({
        "id": 1,
        "url": "None",
        "name": "zhangyp"
    })

    dataset['categories'] = []
    dataset['images'] = []
    dataset['annotations'] = []

    # build the category based on the depth
    classes = list(range(1, 257))
    # classes = np.array(np.linspace(1*cm, 3*cm, 256))

    for cls in classes:
        dataset['categories'].append({'id': int(cls), 'name': str(cls), 'supercategory': 'Depth'})

    # images = np.sort(os.listdir('/Users/zhangyunping/PycharmProjects/Holo_synthetic/test_data/hologram'))
    crop_w, crop_h = 512, 512
    stride = 256
    IMG_ID = 0
    ANNO_CNT = 0

    for n in range(0,hologram_number):
        t1 = time.time()
        # generate the random 3D location
        NUMBER = np.random.randint(low=particle_number[0], high=particle_number[1], dtype=int)
        Z_list = np.array(np.linspace(depth_range[0], depth_range[1], 256))
        particles = particle_field(NUMBER,xyrange=9*mm,z_list=Z_list,size_range=size_range)
        particles = particles.sort_values(by=['z'],ascending=False)
        particles.to_csv("test/param/%d.csv"% n, index=False)
        holo = generate_holo_fromcsv2("test/param/%d.csv"% n)
        plt.imsave("test/img_orignal/%d.jpg" % n, holo, cmap='gray')

        # split into subimages
        idx = 0
        for c_step in range(int((N - crop_h) // stride + 1)):
            for r_step in range(int((N - crop_w) // stride + 1)):
                boundary = [int(stride * r_step), int(stride * r_step + crop_w), int(stride * c_step),
                        int(stride * c_step + crop_h)]
                # boundary = [x0,x1,y0,y1]
                if boundary[3] <= N and boundary[1] <= N:
                    # print(bbox)
                    clipped_img = holo[boundary[2]:boundary[3], boundary[0]:boundary[1]]
                else:
                    continue

                # get the clipped bbox label
                check_cnt = ANNO_CNT
                no_particle = False
                for (p_x,p_y,p_z,p_s) in particles.values:
                    old_bbox = get_bbox(p_x,p_y,p_z,p_s)[0:4]
                    new_bbox = get_new_bbox(old_bbox, boundary)

                    if len(new_bbox) != 4:
                        continue

                    category_id = int((p_z - 1 * cm) / res_z) + 1
                    if category_id == 256:
                        category_id = 255
                    dataset['annotations'].append({
                        'area': new_bbox[2] * new_bbox[3],
                        'bbox': new_bbox,
                        'category_id': category_id,
                        'id': ANNO_CNT,
                        'image_id': IMG_ID,
                        'iscrowd': 0,
                    })
                    ANNO_CNT += 1


                if ANNO_CNT == check_cnt:
                    #if subimage has no particle in it, do not save the image
                    no_particle = True

                # save the cropped image
                if not no_particle:
                    name = str(n) + '_' + str(idx) + '.png'
                    clipped_img = Image.fromarray((clipped_img / np.max(clipped_img) * 255).astype(np.uint8)).convert(
                        'RGB')
                    clipped_img.save('test/img_cropped' + '/' + name)
                    dataset['images'].append(
                        ({'id': IMG_ID, 'width': crop_w, 'height': crop_h, 'file_name': name, 'license': 'None'}))
                    IMG_ID += 1
                    idx += 1



    import json
    json_name = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/test/test.json'
    with open(json_name,'w') as f:
        json.dump(dataset,f)



    # coco = COCO(annotation_file="/Users/zhangyunping/PycharmProjects/Holo_synthetic/test/test.json")
    # # coco = COCO(annotation_file='/Users/zhangyunping/PycharmProjects/Holo_synthetic/comparison/annotations_clip_512_fortest.json')
    # img_ids = coco.getImgIds()
    # for i in range(12,15):
    #     annotation_ids = coco.getAnnIds(imgIds=img_ids[i])
    #     annos = coco.loadAnns(annotation_ids)
    #     image_info = coco.loadImgs(img_ids[i])
    #     image_path = image_info[0]["file_name"]
    #     image_path = os.path.join("/Users/zhangyunping/PycharmProjects/Holo_synthetic/test/img_cropped", image_path)
    #     # image_path = os.path.join("/Users/zhangyunping/PycharmProjects/Holo_synthetic/comparison/images/test_01", image_path)
    #
    #     print("image path (crowd label)", image_path)
    #     image = Image.open(image_path)
    #     draw = ImageDraw.Draw(image)
    #     for ann in annos:
    #         bbox = ann['bbox']
    #         x1 = int(bbox[0])
    #         y1 = int(bbox[1])
    #         x2 = int(bbox[0]+bbox[2])
    #         y2 = int(bbox[1]+bbox[3])
    #         label_name = str(ann['category_id'])
    #         draw.rectangle([x1, y1, x2, y2], outline='red')
    #         draw.text((x1, y1), label_name, (0, 255, 255))
    #     image.show()
