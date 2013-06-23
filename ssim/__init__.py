"""
This module computes the Structured Similarity Image Metric (SSIM)

Created on 21 nov. 2011
@author: Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr, http://isit.u-clermont1.fr/~anvacava

Modified by Christopher Godfrey, on 17 July 2012 (lines 32-34)
Modified by Jeff Terrace, starting 29 August 2012
"""

import numpy as np
import scipy.ndimage
from numpy.ma.core import exp
from scipy.constants.constants import pi
import numexpr as ne

def _to_grayscale(bgr_image):
    flat_image = bgr_image.reshape((-1, 3)).astype('uint32')

    r = flat_image[:, 2]
    g = flat_image[:, 1]
    b = flat_image[:, 0]

    luma = (r * 299 + g * 587 + b * 114) / 1000
    luma = luma.astype('float')

    return luma.reshape((bgr_image.shape[0], bgr_image.shape[1]))

def create_gaussian_kernel(gaussian_kernel_sigma = 1.5, gaussian_kernel_width = 11):
    #Gaussian kernel definition
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))
    mu = int(gaussian_kernel_width / 2)
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = \
                (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                exp(-(((i - mu) ** 2) + ((j - mu) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    return gaussian_kernel

def compute_ssim(im1, im2, gaussian_kernel = None):
    """
    The function to compute SSIM
    @param im1: numpy image
    @param im2: numpy image
    @return: SSIM float value
    """

    if gaussian_kernel == None:
        gaussian_kernel = create_gaussian_kernel()

    # convert the images to grayscale
    img_mat_1 = _to_grayscale(im1)
    img_mat_2 = _to_grayscale(im2)
    
    #Squares of input matrices
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)
    
    #Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)
    
    #Centered squares of variances
    img_mat_sigma_1_sq -= img_mat_mu_1_sq
    img_mat_sigma_2_sq -= img_mat_mu_2_sq
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12
    
    #set k1,k2 & c1,c2 to depend on L (width of color map)
    l = 255
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2
    
    ssim_map = ne.evaluate("(2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2) / \
                ((img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
                 (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2))")
    index = np.average(ssim_map)

    return index
