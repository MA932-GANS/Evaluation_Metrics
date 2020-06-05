#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os
import mahotas as mt
import matplotlib.pyplot as plt
import numbers
from scipy import stats
import matplotlib.offsetbox as offsetbox



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

    files = os.listdir(path)

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
                   "Homogeneity" : Inverse_Difference_Moment,
                   "Sum Average" : Sum_Average,
                   "Sum Variance" : Sum_Variance,
                   "Sum Entropy" : Sum_Entropy,
                   "Entropy" : Entropy,
                   "Difference Variance" : Difference_Variance,
                   "Difference Entropy" : Difference_Entropy,
                   "Information Measure of Correlation 1" : Information_Measure_of_Correlation_1,
                   "Information Measure of Correlation 2" : Information_Measure_of_Correlation_2}


def qqplot(x, y, quantiles=None, interpolation='nearest', ax=None, rug=False,
           rug_length=0.05, rug_kwargs=None, **kwargs):
 
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()

    if quantiles is None:
        quantiles = min(len(x), len(y))

    # Compute quantiles of the two samples
    if isinstance(quantiles, numbers.Integral):
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
    else:
        quantiles = np.atleast_1d(np.sort(quantiles))
    x_quantiles = np.quantile(x, quantiles, interpolation=interpolation)
    y_quantiles = np.quantile(y, quantiles, interpolation=interpolation)

    # Draw the rug plots if requested
    if rug:
        # Default rug plot settings
        rug_x_params = dict(ymin=0, ymax=rug_length, c='gray', alpha=0.5)
        rug_y_params = dict(xmin=0, xmax=rug_length, c='gray', alpha=0.5)

        # Override default setting by any user-specified settings
        if rug_kwargs is not None:
            rug_x_params.update(rug_kwargs)
            rug_y_params.update(rug_kwargs)

        # Draw the rug plots
        for point in x:
            ax.axvline(point, **rug_x_params)
        for point in y:
            ax.axhline(point, **rug_y_params)

    # Draw the q-q plot
    ax.scatter(x_quantiles, y_quantiles, **kwargs)
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
   

    
    
def QQandBox(real,fake,feature):
    """ Q-Q plot and Boxplot next to each other for one feature with p-value"""
    
    p = stats.ks_2samp(real[feature], fake[feature])[1]
    
    left = 0.2
    bottom = 0.2
    top = 0.8
    right = 0.8
    main_ax = plt.axes([left,bottom,right-left,top-bottom])
    main_ax.set_xlabel('Real image data')
    main_ax.set_ylabel('Generated image data')
    main_ax.set_title(feature)
    # create axes to the top and right of the main axes and hide them
    right_ax = plt.axes([right,bottom,1-right,top-bottom])
    

    qqplot(real[feature],fake[feature],ax = main_ax, c='r', alpha=0.5, edgecolor='k')
    main_ax.spines['right'].set_visible(False)
    main_ax.spines['top'].set_visible(False)
    # Save the default tick positions, so we can reset them..
    
    right_ax.boxplot(real[feature], positions=[-0.2], widths=1, labels=['Real'],showfliers=False)
    right_ax.boxplot(fake[feature], positions=[1.2], widths=1, labels=['Fake'],showfliers=False)
    
    # set the limits to the box axes
    right_ax.set_xlim(-1,2)
    right_ax.spines['right'].set_visible(False)
    right_ax.spines['top'].set_visible(False)
    right_ax.spines['left'].set_visible(False)
    right_ax.spines['bottom'].set_visible(False)
    plt.setp(right_ax.get_yticklabels(), visible=False)
    right_ax.tick_params(axis='both', which='both', length=0)
    
    if p <0.0001:
    
        text = 'p = %e' %p
    else:
        text = 'p = %0.5f' %p
    
    ob = offsetbox.AnchoredText(text, loc=4,
                    prop=dict(color='black', size=10))
    ob.patch.set(boxstyle='round', color='white', alpha=0.5)
    main_ax.add_artist(ob)
    plt.savefig('%s_QQandBoxplots.png' %feature)











