import argparse
import cv2
import torch
from utils.general import xyxy2xywh
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt-path', type=str, default='/data/data_server/jju/datasets/FACE/widerface/widerface/train/label.txt', help='widerface label.txt')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parser()

    txt_path = opt.txt_path
    imgs_path = []
    words = []

    f = open(txt_path, 'r')
    lines = f.readlines()
    isFirst = True
    labels = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if isFirst is True:
                isFirst = False
            else:
                labels_copy = labels.copy()
                words.append(labels_copy)
                labels.clear()
            path = line[2:]
            path = txt_path.replace('label.txt','images/') + path
            imgs_path.append(path)
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)

    words.append(labels)
    print(len(imgs_path))  
    print(len(words))
    print(imgs_path[0])
    exit()
    for img, labels in zip(imgs_path, words):
        
        img0 = cv2.imread(path)  # BGR
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        print(gn)
        
        for l in labels:            
            xyxy = torch.Tensor([l[0], l[1], l[0] + l[2], l[1] + l[3]]).view(1,4) # x1y1x2y2        
            xywh = xyxy2xywh(xyxy / gn)
        xyxy_bboxes = torch.cat(xyxy_bboxes, dim=0)
        
        
        
                        
            
            