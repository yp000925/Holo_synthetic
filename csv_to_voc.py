# import os
# import parser
# import csv
# import cv2
# from lxml.etree import Element, SubElement, tostring
# from xml.dom.minidom import parseString
# import shutil
#
# def check_make_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#
# def make_voc_dir(root):
#     check_make_dir(os.path.join(root, 'VOC2007/ImageSets'))
#     check_make_dir(os.path.join(root, 'VOC2007/ImageSets/Main'))
#     check_make_dir(os.path.join(root, 'VOC2007/JPEGImages'))
#     check_make_dir(os.path.join(root, 'VOC2007/Annotations'))
#
# if __name__ == "__main__":
#
#     ROOT = "/PATH/TO/YOUR/GENERATED/DATASET/"
#     img_path = "/IMAGE/SOURCE/DIR/"
#     make_voc_dir(ROOT)
#
# #   create two dict, use file names as keys.
#     obj_label = {}
#     obj_name = {}
#     FOR ALL IMAGE:
#         FOR ALL BOX:
#             obj_name[f'{i}.jpg'].append(single_label['label'])
#             obj_label[f'{i}.jpg'].append({'xmin': min(geo[0], geo[2]), 'ymin': min(geo[1], geo[3]),
#                                  'xmax': max(geo[0], geo[2]), 'ymax': max(geo[1], geo[3])})
#     for img_file in obj_label:
#         image_path = os.path.join(img_path, img_file)
#         img = cv2.imread(image_path)
#         # print(img_path)
#         height, width, channel = img.shape
#         new_img_path = os.path.join(ROOT, 'VOC2007/JPEGImages')
#         shutil.copy(ima除了ge_path, os.path.join(new_img_path, img_file))
#         node_root = Element('annotation')
#         node_folder = SubElement(node_root, 'folder')
#         node_folder.text = 'JPEGImages'
#         node_filename = SubElement(node_root, 'filename')
#         node_filename.text = os.path.basename(image_path)
#         node_size = SubElement(node_root, 'size')
#         node_width = SubElement(node_size, 'width')
#         node_width.text = '%s' % width
#         node_height = SubElement(node_size, 'height')
#         node_height.text = '%s' % height
#         node_depth = SubElement(node_size, 'depth')
#         node_depth.text = '%s' % channel
#
#         for class_name, obj in zip(obj_name[img_file], obj_label[img_file]):
#             xmin, ymin, xmax, ymax = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
#
#             node_object = SubElement(node_root, 'object')
#             node_name = SubElement(node_object, 'name')
#             node_name.text = class_name
#             node_difficult = SubElement(node_object, 'difficult')
#             node_difficult.text = '0'
#             node_bndbox = SubElement(node_object, 'bndbox')
#             node_xmin = SubElement(node_bndbox, 'xmin')
#             node_xmin.text = '%s' % xmin
#             node_ymin = SubElement(node_bndbox, 'ymin')
#             node_ymin.text = '%s' % ymin
#             node_xmax = SubElement(node_bndbox, 'xmax')
#             node_xmax.text = '%s' % xmax
#             node_ymax = SubElement(node_bndbox, 'ymax')
#             node_ymax.text = '%s' % ymax
#             node_name = SubElement(node_object, 'pose')
#             node_name.text = 'Unspecified'
#             node_name = SubElement(node_object, 'truncated')
#             node_name.text = '0'
#         xml = tostring(node_root, pretty_print=True)  # 'annotation'
#         dom = parseString(xml)
#         # save_dir = 'VOC2007/Annotations'
#         xml_name = img_file.replace('.jpg', '.xml')
#         xml_path = ROOT + 'VOC2007/Annotations/' + xml_name
#         with open(xml_path, 'wb') as f:
#             f.write(xml)
#
#
# import os
# import random
#
# trainval_percent = 0.8
# train_percent = 0.7
# xmlfilepath = '/PATH/TO/VOC2007/Annotations'
# txtsavepath = '/PATH/TO/VOC2007/ImageSets/Main'
# total_xml = os.listdir(xmlfilepath)
#
# num=len(total_xml)
# list=range(num)
# tv=int(num*trainval_percent)
# tr=int(tv*train_percent)
# trainval= random.sample(list,tv)
# train=random.sample(trainval,tr)
#
# ftrainval = open(txtsavepath+'/trainval.txt', 'w')
# ftest = open(txtsavepath+'/test.txt', 'w')
# ftrain = open(txtsavepath+'/train.txt', 'w')
# fval = open(txtsavepath+'/val.txt', 'w')
#
# for i  in list:
#     name=total_xml[i][:-4]+'\n'
#     if i in trainval:
#         ftrainval.write(name)
#         if i in train:
#             ftrain.write(name)
#         else:
#             fval.write(name)
#     else:
#         ftest.write(name)
#
# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest .close()