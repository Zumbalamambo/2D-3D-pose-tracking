#!/usr/bin/env python
from config import cfg
from modeling.afm import AFM
import cv2
import numpy as np
import argparse
import os
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch Line Segment Detection (Testing)')

    # parser.add_argument("--img", type=str, required=True)

    parser.add_argument("--config-file",
        metavar = "FILE",
        help = "path to config file",
        type=str,
        required=True,
    )

    parser.add_argument("--gpu", type=int, default = 0)

    parser.add_argument("--epoch", dest="epoch",default=-1, type=int)


    parser.add_argument("opts",
        help="Modify config options using the command-line",
        default = None,
        nargs = argparse.REMAINDER
    )

    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # camera parameters
    # mtx=np.array([[cfg.projection_parameters.fx, 0, cfg.projection_parameters.cx],
    # [0, cfg.projection_parameters.fy, cfg.projection_parameters.cy],
    # [0,0,1]])
    # dist=np.array([cfg.distortion_parameters.k1,cfg.distortion_parameters.k2,cfg.distortion_parameters.p1, cfg.distortion_parameters.p2])

    # image=cv2.imread(args.img)
    # if len(image.shape) == 2:  # gray image
    #     img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    # else:
    #     img = image.copy()

    # dst_img=cv2.undistort(img, mtx,dist)
    system = AFM(cfg)
    system.test(cfg, args.epoch)
    # system.detect(dst_img, cfg)

    
