# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy
import os

def first_filter(img):
    #Mean Filter
    img_Blur=cv2.blur(img,(5,5))
    # Gaussian filter
    img_GaussianBlur=cv2.GaussianBlur(img,(7,7),0)
    # Gaussian bilateral filter
    img_bilateralFilter=cv2.bilateralFilter(img,40,75,75)
    return img, img_Blur

########################
#Edge Detection
def edge_detection(img):
    #img = cv2.imread(file, 0)
    #img = cv2.imread("01.jpg", 0)
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)# 转回uint8
    absY = cv2.convertScaleAbs(y)
    img_edge = cv2.addWeighted(absX,0.5,absY,0.5,0)
    
    #cv2.imshow("absX", absX)
    #cv2.imshow("absY", absY)
    #cv2.imshow("Result", img_edge)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    fig = plt.figure(figsize = (30, 30))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    #ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(img_edge, cmap = plt.cm.gray)
    plt.show()
    
    return img, img_edge

############################################
#Pixel Binarization
def pixel_polarization(img_edge, img, threshold): # threshold 像素两极化的阈值
    for i in range(len(img_edge)):
        for j in range(len(img_edge[i,:])):
            if img_edge[i][j] > threshold:
                img_edge[i][j] = 255
            else:
                img_edge[i][j] = 0
    '''
    fig = plt.figure(figsize = (16, 16))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(img_edge, cmap = plt.cm.gray)
    plt.show()
    '''
    img_edge_polar = img_edge
    return img_edge_polar


def positioning_middle_point(img, dst, point_pixel):
    h, w = img.shape
    w1 = w // 5  # x coordinate of left vertical bar
    w2 = (w // 5) * 4 # x coordinate of right vertical bar
    '''
    print("roi width: ",h, w1, w2)
    '''
    low_l = False
    high_l = False
    while (not low_l or not high_l) and w1 < (w // 2):
        for i, pix in enumerate(dst[:, w1]):
            if i+1 < (h // 2) and not low_l:
                if pix == 255:
                    low_l = True
                    lower_left = i
            elif i+1 > (h // 2) and not high_l:
                h_h = int(h * (3/2) - (i+1)) # Division will bring decimals, so use int(), h/2 to start symmetrical positions to find bright spots
                '''
                print(h_h)
                '''
                if dst[h_h, w1] == 255:
                    high_l = True
                    higher_left = h_h
        if not low_l or not high_l:
            w1 = w1 + 2
    middle_left = (lower_left + higher_left) // 2

    low_r = False
    high_r = False
    while (not low_r or not high_r) and w2 > (w // 2):
        for i, pix in enumerate(dst[:, w2]):
            if i+1 < (h // 2) and not low_r:
                if pix == 255:
                    low_r = True
                    lower_right = i
            elif i+1 > (h // 2) and not high_r:
                h_h = int(h * (3/2) - (i+1))
                if dst[h_h, w2] == 255:
                    high_r = True
                    higher_right = h_h
        if not low_r or not high_r:
            w2 = w2 - 2
    middle_right = (lower_right + higher_right) // 2
    
    dst[middle_left, w1] = point_pixel
    dst[middle_left+1, w1] = point_pixel
    dst[middle_left-1, w1] = point_pixel
    dst[middle_left, w1 + 1] = point_pixel
    dst[middle_left, w1 - 1] = point_pixel
    dst[middle_right, w2] = point_pixel
    dst[middle_right+1, w2] = point_pixel
    dst[middle_right-1, w2] = point_pixel
    dst[middle_right, w2 + 1] = point_pixel
    dst[middle_right, w2 - 1] = point_pixel

    fig = plt.figure(figsize = (16, 16))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(dst, cmap = plt.cm.gray)
    plt.show()
    
    return dst, middle_left, middle_right, w1, w2

#################################
# Rotation Correction
def rotation_correction(img, dst, middle_right, middle_left, w1, w2):
    tangent_value = float(middle_right - middle_left) / float(w2 - w1)
    rotation_angle = np.arctan(tangent_value)/math.pi*180
    (h,w) = img.shape
    center = (w // 2,h // 2)
    M = cv2.getRotationMatrix2D(center,rotation_angle,1)# Rotation scaling matrix: (rotation center, rotation angle, scaling factor)
    rotated_dst = cv2.warpAffine(dst,M,(w,h))
    rotated_img = cv2.warpAffine(img,M,(w,h))
    fig = plt.figure(figsize = (16, 16))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(rotated_dst, cmap = plt.cm.gray)
    ax3.imshow(rotated_img, cmap = plt.cm.gray)
    plt.show()
    return rotated_dst, rotated_img


def roi(rotated_img, rotated_edge, w1, w2, url):
    h, w = rotated_edge.shape
    r = range(0, h)
    r1 = range(0, h // 2)
    r2 = range(h // 2, h - 1)
    c = range(0, w)
    c1 = range(0, w // 2)
    c2 = range(w // 2, w-1)

    highest_edge = (rotated_edge[r1][:,c].sum(axis=1).argmax())
    lowest_edge = (rotated_edge[r2][:,c].sum(axis=1).argmax() + (h // 2))
    leftest_edge = (rotated_edge[r][:,c1].sum(axis=0).argmax())
    rightest_edge = (rotated_edge[r][:,c2].sum(axis=0).argmax() + (w // 2))
    leftest_edge = w1
    rightest_edge = w2
    _, img_w = rotated_edge.shape
    half = int(img_w/2)
    max_right_sum = 0
    max_right_i = 0
    sum_img = numpy.sum(rotated_img,axis=0)
    for i in range(half,img_w-50):
        s = sum(sum_img[i:i+50])
        if s > max_right_sum:
            max_right_sum = s
            max_right_i = i

    #print(highest_edge, lowest_edge, leftest_edge, rightest_edge)
    #print max_right_i
    #rightest_edge = max_right_i + 200
    #leftest_edge = 0
    rotated_edge[highest_edge, : ] = 200
    rotated_edge[lowest_edge, : ] = 200 #150
    rotated_edge[: , leftest_edge] = 200 #200
    rotated_edge[: , rightest_edge] = 200 #250
    rotated_croped = rotated_edge[highest_edge : lowest_edge, leftest_edge : rightest_edge]
    rotated_croped_img = rotated_img[highest_edge : lowest_edge, leftest_edge : rightest_edge]
    fig = plt.figure(figsize = (30, 30))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.imshow(rotated_edge, cmap = plt.cm.gray)
    ax2.imshow(rotated_croped, cmap = plt.cm.gray)
    ax3.imshow(rotated_img, cmap = plt.cm.gray)
    ax4.imshow(rotated_croped_img, cmap = plt.cm.gray)
    plt.show()
    #print("rotated_croped_img type: ", rotated_croped_img)
    #cv2.imwrite(url, rotated_croped_img)

    #im = Image.fromarray(rotated_croped_img)
    #im.save(url)
    return rotated_croped_img

def img_resized_enhance(img, url):
    # Scale Normalization
    #resized_img = cv2.resize(img, (136, 100), cv2.INTER_NEAREST) #最近邻插值
    resized_img = cv2.resize(img, (320, 240), cv2.INTER_LINEAR) #双线性插值
    #resized_img = cv2.resize(img, (136, 100), cv2.INTER_NEAREST) #最近邻插值
    
    fig = plt.figure(figsize = (30, 20))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img, cmap = plt.cm.gray)
    ax2.imshow(resized_img, cmap = plt.cm.gray)
    plt.show()
    
    norm_resized_img = resized_img
    # Grayscale Normalization
    norm_resized_img = cv2.normalize(resized_img, norm_resized_img, 0, 255, cv2.NORM_MINMAX)
    # Histogram Equalization
    #equ_resized_img = cv2.equalizeHist(resized_img)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_resized_img = clahe.apply(norm_resized_img)
    
    plt.figure(figsize = (30, 30))
    plt.subplot(2, 2, 1), plt.title('image')
    plt.imshow(img, cmap = plt.cm.gray)
    plt.subplot(2, 2, 2), plt.title('resized_img')
    plt.imshow(resized_img, cmap = plt.cm.gray)
    plt.subplot(2, 2, 3), plt.title('norm_resized_img')
    plt.imshow(norm_resized_img, cmap = plt.cm.gray)
    plt.subplot(2, 2, 4), plt.title('CLAHE')
    plt.imshow(clahe_resized_img, cmap = plt.cm.gray)
    plt.show()
    
    print('saving...')
    #Create a Folder to save
    cv2.imwrite(url, clahe_resized_img)
    print('done')
    return clahe_resized_img

def get_imgs_roi(img_file):
    images = os.listdir(img_file)
    for i, image in enumerate(images):
        print(i)
        print(image)
        img_raw = cv2.imread(os.path.join(img_file, image), 0)
        print(img_raw.shape)
       
        (h,w) = img.shape
        center = (w / 2,h / 2)
        M = cv2.getRotationMatrix2D(center,90,1)#Rotation scaling matrix: (rotation center, rotation angle, scaling factor)
        img_raw = cv2.warpAffine(img,M,(w,h))
        
        #img_raw, img_edge = edge_detection(img_raw)
        img_raw, img_Blur = first_filter(img_raw)
        img_raw, img_Blur_edge = edge_detection(img_Blur)
        
        fig = plt.figure(figsize = (50, 15))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax1.imshow(img_raw, cmap = plt.cm.gray)
        ax2.imshow(img_edge, cmap = plt.cm.gray)
        ax3.imshow(img_Blur_edge, cmap = plt.cm.gray)
        plt.show()
        
        img_Blur_edge_polar = pixel_polarization(img_Blur_edge, img_raw, 25) # Binarization
        img_Blur_edge_polar_midd, middle_left, middle_right, w1, w2= positioning_middle_point(img_raw, img_Blur_edge_polar, 100)
        img_Blur_edge_polar_midd_rotated, rotated_img = rotation_correction(img_raw, img_Blur_edge_polar_midd, middle_right, middle_left, w1, w2)
        # roi image save path
        new_file = './roi_600_2_all_320240'
        save_root = os.path.join(new_file,image)
        roi_img = roi(rotated_img, img_Blur_edge_polar_midd_rotated, w1, w2, save_root)
        resized_roi_img = img_resized_enhance(roi_img, save_root)


def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4):
        params = {'ksize':(ksize, ksize), 'sigma':3.3, 'theta':theta, 'lambd':18.3,
                  'gamma':4.5, 'psi':0.89, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def getGabor(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum
