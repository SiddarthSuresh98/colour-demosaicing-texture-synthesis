# This code is part of:
#
#   CMPSCI 670: Computer Vision, Spring 2024
#   University of Massachusetts, Amherst
#   Instructor: Grant Van Horn

import numpy as np


def demosaicImage(image, method):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    elif method.lower() == 'linear':
        return demosaicLinear(image.copy()) # Implement this
    elif method.lower() == 'adagrad':
        return demosaicAdagrad(image.copy()) # Implement this
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[1:image_height:2, 1:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 0] = img[1:image_height:2, 1:image_width:2]

    blue_values = img[0:image_height:2, 0:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 2] = img[0:image_height:2, 0:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape
    for i in range(image_height):
        for j in range(image_width):
            if(i%2==0 and j%2==0):
                #blue color pixel 
                if(i+1<image_height):
                    #green value
                    mos_img[i, j, 1] = img[i+1,j]
                    if(j+1<image_width):
                        #red value
                        mos_img[i, j, 0] = img[i+1,j+1]
                    else:
                        mos_img[i,j,0] = img[i+1,j-1]
                else:
                    #green value
                    mos_img[i, j, 1] = img[i-1,j]
                    if(j+1<image_width):
                        #blue value
                        mos_img[i, j, 0] = img[i-1,j+1]
                    else:
                        mos_img[i,j,0] = img[i-1,j-1]
                #blue value
                mos_img[i,j,2] = img[i,j]
            
            elif(i%2==1 and j%2==0):
                #even row odd col
                #green pixel
                if(i+1<image_height):
                    #blue value
                    mos_img[i, j, 2] = img[i+1,j]
                else:
                    #blue value
                    mos_img[i, j, 2] = img[i-1,j]
                if(j+1<image_width):
                    #red value
                    mos_img[i, j, 0] = img[i,j+1]
                else:
                    #red value
                    mos_img[i,j,0] = img[i,j-1]
                #green value
                mos_img[i,j,1] = img[i,j]
            
            elif(i%2==1 and j%2==1):
                #red pixel
                if(i+1<image_height):
                    #blue value
                    if(j+1<image_width):
                        mos_img[i,j,2] = img[i+1,j+1]
                    else:
                        mos_img[i,j,2] = img[i+1,j-1]
                    #green value
                    mos_img[i,j,1] = img[i+1,j]
                else:
                    #blue value
                    if(j+1<image_width):
                        mos_img[i,j,2] = img[i-1,j+1]
                    else:
                        mos_img[i,j,2] = img[i-1,j-1]
                    #green value
                    mos_img[i,j,1] = img[i-1,j]
                #red value
                mos_img[i,j,0] = img[i,j]

            else:
                #green pixel
                if(i+1<image_height):
                    #blue value
                    mos_img[i, j, 0] = img[i+1,j]
                else:
                    #blue value
                    mos_img[i, j, 0] = img[i-1,j]
                if(j+1<image_width):
                    #red value
                    mos_img[i, j, 2] = img[i,j+1]
                else:
                    mos_img[i,j,2] = img[i,j-1]
                #green value
                mos_img[i,j,1] = img[i,j]


    return mos_img


def demosaicLinear(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape
    for i in range(image_height):
        for j in range(image_width):
            sum_green = []
            sum_red = []
            sum_blue = []

            sum_green.clear()
            sum_red.clear()
            sum_blue.clear()

            if(i%2==0 and j%2==0):
                #blue color pixel 
                #green value
                if(j-1>=0):
                    sum_green.append(img[i,j-1])
                if(i-1>=0):
                    sum_green.append(img[i-1,j])
                if(i+1<image_height):
                    sum_green.append(img[i+1,j])
                if(j+1<image_width):
                    sum_green.append(img[i,j+1])
                mos_img[i,j,1] = np.mean(np.array(sum_green))
                
                #red value
                if(i-1>=0):
                    if(j-1>=0):
                        sum_red.append( img[i-1,j-1])
                    if(j+1<image_width):
                        sum_red.append( img[i-1,j+1])
                if(i+1<image_height):
                    if(j-1>=0):
                        sum_red.append( img[i+1,j-1])
                    if(j+1<image_width):
                        sum_red.append(img[i+1,j+1])
                mos_img[i,j,0] = np.mean(np.array(sum_red))
                
                #blue value
                mos_img[i,j,2] = img[i,j]
            
            elif(i%2==1 and j%2==1):
                #red color pixel
                #green value
                if(j-1>=0):
                    sum_green.append( img[i,j-1])
                if(i-1>=0):
                    sum_green.append( img[i-1,j])
                if(i+1<image_height):
                    sum_green.append( img[i+1,j])
                if(j+1<image_width):
                    sum_green.append( img[i,j+1])
                mos_img[i,j,1] = np.mean(np.array(sum_green))

                #blue value
                if(i-1>=0):
                    if(j-1>=0):
                        sum_blue.append( img[i-1,j-1])
                    if(j+1<image_width):
                        sum_blue.append( img[i-1,j+1])
                if(i+1<image_height):
                    if(j-1>=0):
                        sum_blue.append( img[i+1,j-1])
                    if(j+1<image_width):
                        sum_blue.append( img[i+1,j+1])
                mos_img[i,j,2] = np.mean(np.array(sum_blue))

                #red value
                mos_img[i,j,0] = img[i,j]

            elif(i%2==1 and j%2==0):
                #green color pixel
                #red value
                if(j-1>=0):
                    sum_red.append( img[i,j-1])
                if(j+1<image_width):
                    sum_red.append(img[i,j+1])
                mos_img[i,j,0] = np.mean(np.array(sum_red))
                #blue value
                if(i-1>=0):
                    sum_blue.append( img[i-1,j])
                if(i+1<image_height):
                    sum_blue.append( img[i+1,j])
                mos_img[i,j,2] = np.mean(np.array(sum_blue))
                #green
                mos_img[i,j,1] = img[i,j]

            else:
                #green color pixel
                #blue value
                if(j-1>=0):
                    sum_blue.append(img[i,j-1])
                if(j+1<image_width):
                    sum_blue.append(img[i,j+1])
                mos_img[i,j,2] = np.mean(np.array(sum_blue))
                #red value
                if(i-1>=0):
                    sum_red.append( img[i-1,j])
                if(i+1<image_height):
                    sum_red.append( img[i+1,j])
                mos_img[i,j,0] = np.mean(np.array(sum_red))
                #green
                mos_img[i,j,1] = img[i,j]

    return mos_img


def demosaicAdagrad(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape
    for i in range(image_height):
        for j in range(image_width):
            sum_green_tb = []
            sum_green_lr = []
            sum_red = []
            sum_blue = []

            sum_green_tb.clear()
            sum_green_lr.clear()
            sum_red.clear()
            sum_blue.clear()

            if(i%2==0 and j%2==0):
                #blue color pixel 
                #green value
                if(j-1>=0):
                    sum_green_lr.append(img[i,j-1])
                if(i-1>=0):
                    sum_green_tb.append(img[i-1,j])
                if(i+1<image_height):
                    sum_green_tb.append(img[i+1,j])
                if(j+1<image_width):
                    sum_green_lr.append(img[i,j+1])
                if(len(sum_green_lr)==2 and len(sum_green_tb) ==2):
                    if(np.abs(sum_green_lr[0]-sum_green_lr[1])>np.abs(sum_green_tb[0]-sum_green_tb[1])):
                        mos_img[i,j,1] = np.mean(np.array(sum_green_tb))
                    else:
                        mos_img[i,j,1] = np.mean(np.array(sum_green_lr))
                else:
                    mos_img[i,j,1] = np.mean(np.array(sum_green_lr + sum_green_tb))
                
                #red value
                if(i-1>=0):
                    if(j-1>=0):
                        sum_red.append( img[i-1,j-1])
                    if(j+1<image_width):
                        sum_red.append( img[i-1,j+1])
                if(i+1<image_height):
                    if(j-1>=0):
                        sum_red.append( img[i+1,j-1])
                    if(j+1<image_width):
                        sum_red.append(img[i+1,j+1])
                mos_img[i,j,0] = np.mean(np.array(sum_red))
                
                #blue value
                mos_img[i,j,2] = img[i,j]
            
            elif(i%2==1 and j%2==1):
                #red color pixel
                #green value
                if(j-1>=0):
                    sum_green_lr.append(img[i,j-1])
                if(i-1>=0):
                    sum_green_tb.append(img[i-1,j])
                if(i+1<image_height):
                    sum_green_tb.append(img[i+1,j])
                if(j+1<image_width):
                    sum_green_lr.append(img[i,j+1])
                if(len(sum_green_lr)==2 and len(sum_green_tb) ==2):
                    if(np.abs(sum_green_lr[0]-sum_green_lr[1])>np.abs(sum_green_tb[0]-sum_green_tb[1])):
                        mos_img[i,j,1] = np.mean(np.array(sum_green_tb))
                    else:
                        mos_img[i,j,1] = np.mean(np.array(sum_green_lr))
                else:
                    mos_img[i,j,1] = np.mean(np.array(sum_green_lr + sum_green_tb))

                #blue value
                if(i-1>=0):
                    if(j-1>=0):
                        sum_blue.append( img[i-1,j-1])
                    if(j+1<image_width):
                        sum_blue.append( img[i-1,j+1])
                if(i+1<image_height):
                    if(j-1>=0):
                        sum_blue.append( img[i+1,j-1])
                    if(j+1<image_width):
                        sum_blue.append( img[i+1,j+1])
                mos_img[i,j,2] = np.mean(np.array(sum_blue))

                #red value
                mos_img[i,j,0] = img[i,j]

            elif(i%2==1 and j%2==0):
                #green color pixel
                #red value
                if(j-1>=0):
                    sum_red.append( img[i,j-1])
                if(j+1<image_width):
                    sum_red.append(img[i,j+1])
                mos_img[i,j,0] = np.mean(np.array(sum_red))
                #blue value
                if(i-1>=0):
                    sum_blue.append( img[i-1,j])
                if(i+1<image_height):
                    sum_blue.append( img[i+1,j])
                mos_img[i,j,2] = np.mean(np.array(sum_blue))
                #green
                mos_img[i,j,1] = img[i,j]

            else:
                #green color pixel
                #blue value
                if(j-1>=0):
                    sum_blue.append(img[i,j-1])
                if(j+1<image_width):
                    sum_blue.append(img[i,j+1])
                mos_img[i,j,2] = np.mean(np.array(sum_blue))
                #red value
                if(i-1>=0):
                    sum_red.append( img[i-1,j])
                if(i+1<image_height):
                    sum_red.append( img[i+1,j])
                mos_img[i,j,0] = np.mean(np.array(sum_red))
                #green
                mos_img[i,j,1] = img[i,j]

    return mos_img
