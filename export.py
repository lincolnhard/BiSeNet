import os
import torch
import argparse
from model.build_BiSeNet import BiSeNet
from PIL import Image
from torchvision import transforms
from utils import reverse_one_hot
import numpy as np
import cv2
import traceback
from torchsummary import summary
from thop import profile

def main(params):

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--context_path', type=str, default="resnet18", help='The context path model you are using, resnet18, resnet101.')
    args = parser.parse_args(params)


    model = BiSeNet(args.num_classes, args.context_path)
    model.load_state_dict(torch.load(os.path.join(args.save_model_path, 'best_dice_loss.pth')))
    model.eval()

    img = Image.open('./CamVid/test/Seq05VD_f00660.png')
    transform = transforms.Compose([
        transforms.Resize([720, 960]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img).unsqueeze(dim=0)


    imresult = np.zeros([img.shape[2], img.shape[3], 3], dtype=np.uint8)
    with torch.no_grad():
        predict = model(img).squeeze()
        predict = reverse_one_hot(predict)
        predict = np.array(predict)
        imresult[:,:][predict == 0] = [255, 51, 255]
        imresult[:,:][predict == 1] = [255, 0, 0]
        imresult[:,:][predict == 2] = [0, 255, 0]
        imresult[:,:][predict == 3] = [0, 0, 255]
        imresult[:,:][predict == 4] = [255, 255, 0]
        imresult[:,:][predict == 5] = [255, 0, 255]
        imresult[:,:][predict == 6] = [0, 255, 255]
        imresult[:,:][predict == 7] = [10, 200, 128]
        imresult[:,:][predict == 8] = [125, 18, 78]
        imresult[:,:][predict == 9] = [205, 128, 8]
        imresult[:,:][predict == 10] = [144, 208, 18]
        imresult[:,:][predict == 11] = [5, 88, 198]
    cv2.imwrite('result.png', imresult)
    print('inference done')

    summary(model, (3, 720, 960), 1, "cpu")

    macs, params = profile(model, inputs=(img, ))
    print('macs', macs)
    print('params', params)

    log = open('log.txt', 'w')
    EXPORTONNXNAME = 'nit-bisenet.onnx'
    try:
        torch.onnx.export(model,
                        img,
                        EXPORTONNXNAME,
                        export_params=True,
                        do_constant_folding=True,
                        input_names = ['data'],
                        # output_names = ['output']
                        output_names = ['output']
                        )
    except Exception:
        traceback.print_exc(file=log)

    print('export done')

if __name__ == '__main__':
    params = [
        '--num_classes', '12',
        '--save_model_path', './checkpoints_18_sgd',
    ]
    main(params)