#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:28:25 2023

@author: karan_bania
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import time

def Kmeans(image_path, save_path="peppers-small-compressed", K=16, epsilon=1e-10, print_every=1):
    
    img = imread(image_path)
    
    h = img.shape[0]
        
    new_img = np.ones_like(img, dtype=np.int32) #initialising a new image
    
    centroid_indices = np.random.randint(low=0, high=h, size=(K, 2))
    centroids = np.zeros((K, 3), dtype=np.int32)
    
    for k in range(K):
        centroids[k] = img[centroid_indices[k][0], centroid_indices[k][1]]
        plt.imshow(centroids[k].reshape(1, 1, 3))
        plt.savefig(f"./centroid{k}.png")
        plt.close('all')
        
    colors = np.zeros((h, h), dtype=np.int32)
    cost_fn = np.linalg.norm(img - centroids[0])
    
    iters = 0
    
    while iters<30:
        
        start = time.time()
        
        iters += 1
        
        old_cost = cost_fn
        cost_fn = 0
        for i in range(h):
            for j in range(h):
                min_dist = np.float("inf")
                for k in range(K):
                    dist = np.linalg.norm(img[i][j] - centroids[k])
                    if dist < min_dist:
                        min_dist = dist
                        colors[i][j] = k
                cost_fn += min_dist
                        
        for k in range(K):
            centroids[k] = np.mean(img[colors==k], axis=0).astype(int)
                    
        if np.abs(old_cost - cost_fn)/old_cost < epsilon:
            print(f"last iteration : {iters}, reduction : {np.abs(old_cost - cost_fn)/old_cost}, time : {time.time() - start}s")
            break
        
        elif (iters%print_every == 0):
            print(f"iteration : {iters}, reduction : {np.abs(old_cost - cost_fn)/old_cost}, time : {time.time() - start}s")
            
    for i in range(h):
        for j in range(h):
            new_img[i][j] = (centroids[colors[i][j]])
            
    plt.title("Compressed Image")
    plt.imshow(new_img)
    plt.show()
    plt.savefig(f"./{save_path}.png")
    plt.close('all')