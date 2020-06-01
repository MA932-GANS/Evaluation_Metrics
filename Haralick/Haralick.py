#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os
import mahotas as mt
import matplotlib.pyplot as plt



def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean


def multiple_img_features(path):
    
    """
    Input - A directory containing only all images
    Output - A dictionary with 13 features
        
    """

    files = os.listdir(str(path))

    Angular_Second_Moment = []
    Contrast = []
    Correlation = []
    Variance = []
    Inverse_Difference_Moment = []
    Sum_Average = []
    Sum_Variance = []
    Sum_Entropy = []
    Entropy = []
    Difference_Variance = []
    Difference_Entropy = []
    Information_Measure_of_Correlation_1 = []
    Information_Measure_of_Correlation_2 = []

    for i in files:
        image = cv2.imread(path + i)
        Haral = extract_features(image)
        Angular_Second_Moment.append(Haral[0])
        Contrast.append(Haral[1])
        Correlation.append(Haral[2])
        Variance.append(Haral[3])
        Inverse_Difference_Moment.append(Haral[4])
        Sum_Average.append(Haral[5])
        Sum_Variance.append(Haral[6])
        Sum_Entropy.append(Haral[7])
        Entropy.append(Haral[8])
        Difference_Variance.append(Haral[9])
        Difference_Entropy.append(Haral[10])
        Information_Measure_of_Correlation_1.append(Haral[11])
        Information_Measure_of_Correlation_2.append(Haral[12])
        
    return {"Angular Second Moment" : Angular_Second_Moment,
                   "Contrast" : Contrast,
                   "Correlation" : Correlation,
                   "Sum of Squares: Variance" : Variance,
                   "Inverse Difference Moment" : Inverse_Difference_Moment,
                   "Sum Average" : Sum_Average,
                   "Sum Variance" : Sum_Variance,
                   "Sum Entropy" : Sum_Entropy,
                   "Entropy" : Entropy,
                   "Difference Variance" : Difference_Variance,
                   "Difference Entropy" : Difference_Entropy,
                   "Information Measure of Correlation 1" : Information_Measure_of_Correlation_1,
                   "Information Measure of Correlation 2" : Information_Measure_of_Correlation_2}


def plot_box(real_dict,fake_dict):
    
    """
    Input - Two dictionaries -- Real features , Fake features
    Output - Two boxplots of fearure comparison 
        
    """
    
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,figsize=(14, 10))

    ax1.set_title('Contrast')
    ax1.boxplot([real_dict['Contrast'],fake_dict['Contrast']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax2.set_title('Angular Second Moment')
    ax2.boxplot([real_dict['Angular Second Moment'],fake_dict['Angular Second Moment']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax3.set_title('Correlation')
    ax3.boxplot([real_dict['Correlation'],fake_dict['Correlation']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax4.set_title('Variance')
    ax4.boxplot([real_dict['Sum of Squares: Variance'],fake_dict['Sum of Squares: Variance']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax5.set_title('Inverse Difference Moment')
    ax5.boxplot([real_dict['Inverse Difference Moment'],fake_dict['Inverse Difference Moment']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax6.set_title('Sum Average')
    ax6.boxplot([real_dict['Sum Average'],fake_dict['Sum Average']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    fig.savefig('Haralick_1-6_box_plot.png')
    
    fig2, ((ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(nrows=2, ncols=3,figsize=(14, 10))

    ax7.set_title('Sum Variance')
    ax7.boxplot([real_dict['Sum Variance'],fake_dict['Sum Variance']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax8.set_title('Sum Entropy')
    ax8.boxplot([real_dict['Sum Entropy'],fake_dict['Sum Entropy']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax9.set_title('Entropy')
    ax9.boxplot([real_dict['Entropy'],fake_dict['Entropy']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax10.set_title('Difference Variance')
    ax10.boxplot([real_dict['Difference Variance'],fake_dict['Difference Variance']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax11.set_title('Difference Entropy')
    ax11.boxplot([real_dict['Difference Entropy'],fake_dict['Difference Entropy']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    ax12.set_title('Information Measure of Correlation 1')
    ax12.boxplot([real_dict['Information Measure of Correlation 1'],fake_dict['Information Measure of Correlation 1']],positions=[1, 1.6,],labels=['Ground Truth','Generated'],showfliers=False)
    
    fig2.savefig('Haralick_7-12_box_plot.png')
    
    











