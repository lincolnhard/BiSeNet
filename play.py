import torch
from dataset.CamVid import CamVid
from model.build_BiSeNet import BiSeNet
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

if __name__ == '__main__':
    data = CamVid(['/mnt/data/lincoln/BiSeNet/CamVid/train', '/mnt/data/lincoln/BiSeNet/CamVid/val'],
                  ['/mnt/data/lincoln/BiSeNet/CamVid/train_labels', '/mnt/data/lincoln/BiSeNet/CamVid/val_labels'], '/mnt/data/lincoln/BiSeNet/CamVid/class_dict.csv',
                  (720, 960), loss='dice', mode='train')

    for i, (img, label) in enumerate(data):
        img.save(str(i) + '_im.jpg')
        label.save(str(i) + '_label.jpg')
        # print(label.size())
        # print(img.size())
        # print(torch.max(label))
    