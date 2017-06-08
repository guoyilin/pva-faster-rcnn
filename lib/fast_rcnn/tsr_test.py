#!/usr/bin/env python
# -*- coding:utf-8 -*-
# coding=utf-8
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
大模型测试
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import datetime
#import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import os
CLASSES = ('__background__','number_0','number_1','number_2','number_3','number_4','number_5', 'number_6', 'number_7')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def outputDetectionResult(im, class_name, dets, thresh=0.5):
    outputFile = open('CarDetectionResult_window_30000.txt')
    inds = np.where(dets[:,-1] >= thresh)[0]
    if len(inds) == 0:
        return 


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def runDetection (net, testFileName, detectionFile):
    ftest = open(testFileName,'r')
    timer_all = Timer()
    timer_all.tic()    
    num = 1
   
    for imageFileName in ftest.readlines():
        print 'num:',num
        print 'file_name:',imageFileName
        imageFileName = imageFileName.strip() 
        if(not os.path.exists(imageFileName)):
            continue 
        old_im = cv2.imread(imageFileName)
        if(old_im is None):
		continue
	#resize to 256(according to w/h)
        old_width = old_im.shape[1]
#        shorter_size = old_width
        old_height = old_im.shape[0]
        new_width = 0
        new_height = 0
#        if(old_height < shorter_size):
#            shorter_size = old_height
#            new_height = 600
#            new_width = 600 * float(old_width)/old_height
#        else:
#            shorter_size = old_width
#            new_width = 600
#            new_height = 600 * float(old_height)/old_width
#        im = cv2.resize(old_im, (int(new_width),int(new_height)))
        im = old_im
        new_width = old_width
        new_height = old_height
        if(new_width < 30 or new_height < 30):
            continue
# Detect all object classes and regress object bounds
        print "new width, height:", new_width, new_height
        timer = Timer()
        timer.tic()
	print "start:",datetime.datetime.now()
        scores, boxes = im_detect(net, im)
	print "end:",datetime.datetime.now()
        timer.toc()
        print "time:", timer.total_time
        #print boxes

        
        # Visualize detections for each class
        CONF_THRESHS  = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        NMS_THRESH = 0.5
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
	    CONF_THRESH = CONF_THRESHS[cls_ind-1]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]  
            print 'Detected number size: ', inds.size
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
	        print bbox
                cv2.rectangle(im, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
                font = cv2.FONT_HERSHEY_SIMPLEX	
 #               text = ''
		
		if(cls_ind == 1):#redcircle
			text = '1200001'
		elif(cls_ind == 2):
			text = '1300001'
		elif(cls_ind == 3):
			text = '1100001'
		elif(cls_ind == 4):
			text = '1470000'
		elif(cls_ind == 5):
			text = '1500001'
		elif(cls_ind == 6):
			text = '1500002'
		elif(cls_ind == 7):
			text = '1500003'
		elif(cls_ind == 8):
			text = '1700001'
 		cv2.putText(im,str(text),(int(bbox[0]),int(bbox[1])), font, 0.3,(255,255,255),1)               
		detectionFile.write(imageFileName + " " + text + " 1 tmp tmp tmp tmp (" +  str(int(bbox[0]))  + ", " + str(int(bbox[1])) + ") - (" + str(int(bbox[2])) + ", " + str(int(bbox[3]))  +  ") "  + str(score)  + "\n")
        cv2.imwrite("test_result/" + imageFileName.split("/")[-1].strip(), im)
	print ('Detection one pic need {:.3f}s for '
               '{:d} ').format(timer.total_time, boxes.shape[0])
        num = num+1
    
    timer_all.toc()
    print ('detect pics: {:d} took {:.3f}s ').format(num-1, timer_all.total_time)



def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='faster rcnn所有通道检测')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('prototxt', help='test prototxt')
    parser.add_argument('caffemodel', help='test model')
    parser.add_argument('test_file', help='test file')
    parser.add_argument('detection_file', help='save detection file') 
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    prototxt = args.prototxt
    caffemodel = args.caffemodel
    
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    print prototxt
    detectionFile = open(args.detection_file, 'w')
    runDetection(net, args.test_file, detectionFile)
