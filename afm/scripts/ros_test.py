#!/usr/bin/env python
import rospy
from config import cfg
from modeling.afm import AFM
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from afm.msg import lines2d
import cv2
import numpy as np
import os

class Nodo(object):
    def __init__(self, cfg):
        # Params
        self.image = None
        self.header = None 
        self.br = CvBridge()
        self.pub_img_set=False 
        self.image_topic=cfg.image_topic

        self.system = AFM(cfg)
        self.system.model.eval()
        self.system.load_weight_by_epoch(-1)
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Publishers
        self.pub = rospy.Publisher('Lines2d', lines2d, queue_size=1000)
        if self.pub_img_set:
            self.pub_image = rospy.Publisher('feature_image', Image, queue_size=1000)

        # Subscribers
        rospy.Subscriber(self.image_topic,Image,self.callback)

        # camera parameters
        self.mtx=np.array([[cfg.projection_parameters.fx, 0, cfg.projection_parameters.cx],
        [0, cfg.projection_parameters.fy, cfg.projection_parameters.cy],
        [0,0,1]])
        self.dist=np.array([cfg.distortion_parameters.k1,cfg.distortion_parameters.k2,cfg.distortion_parameters.p1, cfg.distortion_parameters.p2])
        self.width=cfg.width
        self.height=cfg.height
        self.newmtx, self.validpixROI=cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.width,self.height) , 0, (self.width,self.height))
        # print(self.newmtx)

    def callback(self, msg):
        self.header = msg.header
        self.image = self.br.imgmsg_to_cv2(msg)
        
    def start(self,cfg):
        pre_msg_time=rospy.Time(0)
        while not rospy.is_shutdown():
            if self.image is not None:
                if len(self.image.shape) == 2:  # gray image
                    img = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
                else:
                    img = self.image.copy()
                msg_time=self.header.stamp
                if msg_time>pre_msg_time:
                    pre_msg_time=msg_time
                    # dst_img=cv2.undistort(img, self.mtx, self.dist, newCameraMatrix=self.newmtx)
                    dst_img=img
                    feats = self.system.detect(dst_img, cfg)
                    lines2d_msg = lines2d(
                        header=self.header, startx=feats[:, 0], starty=feats[:, 1], endx=feats[:, 2], endy=feats[:, 3])
                    self.pub.publish(lines2d_msg)
                    if self.pub_img_set:
                        feat_imge = dst_img.copy()
                        for i in range(feats.shape[0]):
                            cv2.line(feat_imge, (feats[i, 0], feats[i, 1]),
                                    (feats[i, 2], feats[i, 3]), (0, 0, 255), 2)
                        self.pub_image.publish(self.br.cv2_to_imgmsg(feat_imge, "bgr8"))

            self.loop_rate.sleep()


if __name__ == "__main__":

    rospy.init_node('afm')

    config_file = rospy.get_param('~config_file')
    img_file = rospy.get_param('~image')
    gpu = rospy.get_param('~gpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    cfg.merge_from_file(config_file)
    # print(cfg)
    my_node = Nodo(cfg)
    my_node.start(cfg)

