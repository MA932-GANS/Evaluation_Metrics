#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import heapq as hq
import copy as cp
import imageio
import random
import math
import os
import cv2



def birthday(generated_img_dir,save_dir):

    images = os.listdir(generated_img_dir)
    N = len(images)
    n = int(math.sqrt(N))
    sampled_images = random.sample(images, n)	# a batch of generated samples
    queue = []	# a priority queue maintaining top K most similar pairs
    topK = 5 # keep top
    
    for i in range(n):
        for j in range(i+1, n):
            image1 = cv2.imread(generated_img_dir + sampled_images[i])
            image2 = cv2.imread(generated_img_dir + sampled_images[j])
            # measure similarity in pixel space 
            dist = np.sum((image1- image2)**2)
            if len(queue) == 0 or -1*dist > queue[0][0]:
                hq.heappush(queue, (-1*cp.deepcopy(dist), cp.deepcopy(image1), cp.deepcopy(image2)))
                if len(queue) > topK:
                    hq.heappop(queue)
    
    for idx in range(topK):
     	neg_dist, img1, img2 = hq.heappop(queue)
     	imageio.imwrite(save_dir + 'pair#%d_%f_%d.png'%(idx, -1*neg_dist, 1), (img1+1.)/2)
     	imageio.imwrite(save_dir + 'pair#%d_%f_%d.png'%(idx, -1*neg_dist, 2), (img2+1.)/2)