import os
import argparse
import torch

def convertWF2json(annotation_path, dir_path):
    
    img_files = []
    words = []

    with open(annotation_path) as f:
        while True:
            img_file_name = f.readline().strip()            
            if img_file_name:            
                img_path = os.path.join(dir_path, img_file_name)                 
                img_files.append(img_path)                       
                nbbox = f.readline()                                       
                word = []                     

                if int(nbbox) == 0:                           
                    dummy = f.readline()
                else:
                    for i in range(0, int(nbbox)):                            
                        x1, y1, w, h, _, _, _, _, _, _ = [float(i) for i in f.readline().split()]
                        word.append([x1,y1,w,h])                    
                words.append(word)
            else:
                break
    
    print(len(img_files), len(words))

        

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--annotations-path', default='data/widerface/wider_face_train_bbx_gt.txt')    
    parser.add_argument('-ip', '--dir_path', default='/data/data_server/jju/datasets/FACE/widerface/widerface/train/images')
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = parser()
    
    convertWF2json(opt.annotations_path, opt.dir_path)
    cache = torch.load('data/widerface.cache')  # load
    print(len(cache.keys()))