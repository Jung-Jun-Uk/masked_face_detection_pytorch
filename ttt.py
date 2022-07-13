import torch
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def parse_annotation(annotation_path, label_dict):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    labels = list()    
    for object in root.iter('object'):

        label = object.find('name').text.lower().strip()
        if label not in label_dict.keys():
            label = 'without_mask'

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        labels.append([label_dict[label], xmin, ymin, xmax, ymax])
                
    return labels


def create_pwmfd_dataset(dir_path, label_dict):
    img_files = sorted(glob.glob(dir_path + '/*.jpg'))        
    anno_xml = sorted(glob.glob(dir_path + '/*.xml'))
        
    new_img_files = []
    labels = []
    for img_path, anno_path in zip(img_files, anno_xml):
        l = parse_annotation(anno_path, label_dict)
        new_img_files.append(img_path)
        labels.append(l)
        
    return new_img_files, labels


info_dict = torch.load('info_dict.pth')

img_dir = '/data/data_server/jju/datasets/FACE/PWMFD/PWMFD_Train' 
label_dict = {'without_mask': 0, 'with_mask': 1, 'incorrect_mask' : 2}
img_files, labels = create_pwmfd_dataset(img_dir, label_dict)

num_length =[]
count = [0] * 10
for img_path, label in zip(img_files, labels):
    num_length.append(len(label))
    
    if len(label) > 10:
        print(img_path, len(label))
    """ if len(label) > 10:
        count[-1] += 1
    else:
        count[len(label) - 1] += 1 """





