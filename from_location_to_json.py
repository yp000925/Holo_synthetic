from LightPipes import *
from utils import *
from propagators import ASM
import numpy as np
import pandas as pd
import time
from PIL import Image,ImageDraw
from pycocotools.coco import COCO
import os
from tqdm import tqdm


def get_buffer(z,size):
    z_rate = z/dep_slice
    # size_rate = (size-size_range[0])/(size_range[1]-size_range[0])
    size_rate = 1
    buffer = 20+10*z_rate+5*size_rate
    # buffer = 30
    # p_size = int(size_range[1]/frame * N)
    # buffer = p_size*10*(z_rate*0.6+size_rate*0.4)
    return buffer


def center_to_bbox(px,py,pz,size):
    buffer = get_buffer(pz,size)
    # left-top coordinate
    bbox_x = max(0,px-buffer)
    bbox_y = max(0,py-buffer)
    height = buffer*2
    width = buffer*2
    if bbox_x+width > N:
        width = N-bbox_x
    if bbox_y+height > N:
        height = N-bbox_y
    seg = [bbox_x,bbox_y,bbox_x,bbox_y+height,bbox_x+width,bbox_y+height,bbox_x+width,bbox_y]
    return (bbox_x,bbox_y,width,height,seg)


def particle_field(x,y,z,s):
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df['size'] = s
    return df


from pathlib import Path
import shutil

def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'param', dir / 'img':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

if __name__ == '__main__':
    base_root = '/home/ypzhang/exp/holo_data/512_experimental_data'
    import scipy.io
    import glob
    data_param_path = ""
    param = scipy.io.loadmat(data_param_path)
    wavelength = param['lambda']
    pixel_pitch = param['pps']
    pixel_pitch_cam = param['pps_cam']
    depth_offset = param['z0']
    depth_range= (param['z'].min(), param['z'].max())
    dep_slice = param['z'].shape[1]
    res_z = param['dz']
    N = param['Nx']
    frame = pixel_pitch*N

    import scipy.io
    csv_file_root = ''
    dataset = {}
    dataset['info'] = []
    dataset['licenses'] = []
    dataset['info'].append({'year': "2021", "version": '1',
                            "description": "Experimental data",
                            "contributor": "zhangyp",
                            "url": "None",
                            "date_created": "2021-10-29"})
    dataset['licenses'].append({
        "id": 1,
        "url": "None",
        "name": "zhangyp"
    })

    dataset['categories'] = []
    dataset['images'] = []
    dataset['annotations'] = []

    # build the category based on the depth
    classes = list(range(1, dep_slice+1))

    for cls in classes:
        dataset['categories'].append({'id': int(cls), 'name': str(cls), 'supercategory': 'Depth'})

    # images = np.sort(os.listdir('/Users/zhangyunping/PycharmProjects/Holo_synthetic/test_data/hologram'))
    # crop_w, crop_h = 512, 512
    # stride = 256
    ANNO_CNT = 0

    for csv_file in tqdm(glob.glob(csv_file_root+'/*.csv')):
        particles = pd.read_csv(csv_file)
        img_idx = csv_file.split('.')[0]
        dataset['images'].append(
            ({'id': img_idx, 'width': N, 'height': N, 'file_name': str(img_idx)+'.png', 'license': 'None'}))
        for (p_x, p_y, p_z, p_s) in particles.values:
            bbox = center_to_bbox(p_x, p_y, p_z, p_s)[0:4]
            if len(bbox) != 4:
                continue
            category_id = int((p_z - depth_range[0]) / res_z) + 1
            if category_id == 256:
                category_id = 255
            dataset['annotations'].append({
                'area': bbox[2] * bbox[3],
                'bbox': bbox,
                'category_id': category_id,
                'id': ANNO_CNT,
                'image_id': img_idx,
                'iscrowd': 0,
            })
            ANNO_CNT += 1



    import json

    json_name = base_root +('experimental_data.json')
    if (n + 1) % 100 == 0:
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
    with open(json_name,'w') as f:
        json.dump(dataset,f)

