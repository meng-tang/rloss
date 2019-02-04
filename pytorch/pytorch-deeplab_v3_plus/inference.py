import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable

from mypath import Path
from dataloaders import make_data_loader
from dataloaders.custom_transforms import denormalizeimage
from dataloaders.utils import decode_segmap
from dataloaders import custom_transforms as tr
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.saver import Saver
import time
import multiprocessing

from DenseCRFLoss import DenseCRFLoss

global grad_seg 

def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Inference")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--n_class', type=int, default=21)
    parser.add_argument('--crop_size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    # checking point
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='put the path to checkpoint if needed')
    # rloss options
    parser.add_argument('--rloss_weight', type=float, default=0,
                        metavar='M', help='densecrf loss (default: 0)')
    parser.add_argument('--rloss_scale',type=float,default=1.0,
                        help='scale factor for rloss input, choose small number for efficiency, domain: (0,1]')
    parser.add_argument('--sigma_rgb',type=float,default=15.0,
                        help='DenseCRF sigma_rgb')
    parser.add_argument('--sigma_xy',type=float,default=80.0,
                        help='DenseCRF sigma_xy')

    # input image
    parser.add_argument('--image_path',type=str,default='./misc/test.png',
                        help='input image path')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    print(args)
    
    # Define network
    model = DeepLab(num_classes=args.n_class,
                    backbone=args.backbone,
                    output_stride=16,
                    sync_bn=False,
                    freeze_bn=False)
    
    # Using cuda
    if not args.no_cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        patch_replication_callback(model)
        model = model.cuda()
    
    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {}) best_pred {}"
          .format(args.checkpoint, checkpoint['epoch'], best_pred))
    
    model.eval()
    densecrflosslayer = DenseCRFLoss(weight=1e-8, sigma_rgb=args.sigma_rgb, sigma_xy=args.sigma_xy)
    if not args.no_cuda:
        densecrflosslayer.cuda()
    print(densecrflosslayer)
    composed_transforms = transforms.Compose([
            tr.FixScaleCropImage(crop_size=args.crop_size),
            tr.NormalizeImage(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensorImage()])
    image = composed_transforms(Image.open(args.image_path).convert('RGB')).unsqueeze(0)
    image_cpu = image
    if not args.no_cuda:
        image = image.cuda()
    output = model(image)
    pred = output.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    # Add batch sample into evaluator
    softmax = nn.Softmax(dim=1)
    probs = softmax(output)
    probs = Variable(probs, requires_grad=True)
    croppings = torch.ones(pred.shape).float()
    if not args.no_cuda:
        croppings = croppings.cuda()
        
    # resize output & image & croppings for densecrf
    start = time.time()
    densecrfloss = densecrflosslayer(image_cpu, probs, croppings,args.rloss_scale)
    print('inference time:',time.time()-start)
    print("densecrf loss {}".format(densecrfloss.item()))

    # visualize densecrfloss
    densecrfloss.backward()
    #print (probs.grad.sum())
    #print (reduced_probs.grad.sum())
    #grad_seg = reduced_probs.grad.cpu().numpy()
    
    #"""
    grad_seg = probs.grad.cpu().numpy()
    #print (grad_seg.shape)

    for i in range(args.n_class):
        fig=plt.figure()
        plt.imshow(grad_seg[0,i,:,:], cmap="hot") #vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        plt.savefig('./misc/'+args.image_path.split('/')[-1].split('.')[0]+'_grad_seg_class_' + str(i) +'.png')
        plt.show(block=False)
        plt.close(fig)

    # visualize prediction
    segmap = decode_segmap(pred[0],'pascal')*255
    np.set_printoptions(threshold=np.nan)
    segmap = segmap.astype(np.uint8)
    segimg = Image.fromarray(segmap, 'RGB')
    segimg.save('./misc/'+args.image_path.split('/')[-1].split('.')[0]+'_prediction.png')
    #"""
        
if __name__ == "__main__":
   main()
