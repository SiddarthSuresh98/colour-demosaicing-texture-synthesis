import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import math
import time

# Load images
img = io.imread('../data/texture/D20.png')
#img = io.imread('../data/texture/Texture2.bmp')
#img = io.imread('../data/texture/english.jpg')

def showGrayScaleImage(im):
    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].set_title("Grayscale")

    fig.tight_layout()
    plt.show()

def synthEfrosLeung(img,winsize,outSize):
    start = time.time()
    img_height,img_width,num_channels = np.shape(img)
    output_image = np.zeros((outSize,outSize))

    #convert image to grayscale
    grayscale_ip = rgb2gray(img[:,:,:3])

    #generate random tile to seed
    tile_row = np.random.randint(0, img_height - 3 + 1)
    tile_col = np.random.randint(0, img_width - 3 + 1)
    random_tile = grayscale_ip[tile_row:tile_row + 3, tile_col:tile_col + 3]

    #initialize output image
    #centre of output image
    center_x = output_image.shape[1] // 2
    center_y = output_image.shape[0] // 2
    start_x = center_x - 1
    start_y = center_y - 1
    output_image[start_y:start_y+3, start_x:start_x+3] = random_tile
    

    #padding the image to resolve border issues
    padded_output_img = np.zeros((output_image.shape[0]+winsize, output_image.shape[1]+winsize))
    padded_output_img[winsize//2:output_image.shape[0]+winsize//2, winsize//2:output_image.shape[1]+winsize//2] = output_image

    #structuring element for binary dilation
    structuring_element = np.ones((3,3), dtype=np.uint8)

    binary_mask =  np.zeros(output_image.shape)
    binary_mask[output_image > 0] = 1

    #running until all pixels in output image are filled
    while(not np.all(binary_mask[:,:] > 0)):

        #list of pixels that are unfilled but have filled pixels in neighbourhood
        dilated_image = binary_dilation(binary_mask , structure=structuring_element)
        imageList = dilated_image - binary_mask

        #sorted pixel list of all required pixels based on count of non zero pixels in neighbourhood
        pixelList = []
        pixelList.clear()
        for i in range(imageList.shape[0]):
            for j in range(imageList.shape[1]):
                non0 = 0
                if(imageList[i,j] > 0):
                    non0 = np.count_nonzero(output_image[max(0, i-1):min(output_image.shape[0], i+2), max(0, j-1):min(output_image.shape[1], j+2)])
                    pixelList.append((i,j,non0))
        pixelList = sorted(pixelList, key=lambda x: x[2],reverse = True)
        
        #For each pixel in pixelList
        for item in pixelList:

            #row, column of pixel
            row = item[0]
            column = item[1]

            #output image window dimensions
            top = max(row - winsize // 2, 0)
            bottom = min(row + winsize // 2 + 1, output_image.shape[0])
            left = max(column - winsize // 2, 0)
            right = min(column + winsize // 2 + 1, output_image.shape[1])

            #output image neighbourhood
            output_neighbourhood = output_image[top: bottom, left: right]
            
            #valid mask for current pixel
            mask = np.zeros((output_neighbourhood.shape[0], output_neighbourhood.shape[1]))
            mask[output_neighbourhood>0] = 1

            #ensuring valid mask is of windowSize * windowSize
            new_shape = (winsize,winsize)
            resized_mask = np.zeros(new_shape, dtype=mask.dtype)
            min_shape = tuple(min(mask_shape, new_shape[i]) for i, mask_shape in enumerate(mask.shape))
            slices = tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))
            resized_mask[slices] = mask[slices]

            #flattening to use speed boost provided by numpy
            mask = resized_mask.flatten()
            
            #output window flattened to use speed boost by numpy
            padded_output_window = (padded_output_img[row:row+winsize,column:column+winsize].flatten())

            #Precomputing all valid windows in input image to be scanned, as these do not change, to increase speed of computation.
            windows = (grayscale_ip.shape[0] - winsize + 1) * (grayscale_ip.shape[1] - winsize + 1)
            image_windows = np.zeros((windows, padded_output_window.shape[0]))
            idx = 0
            for i in range(grayscale_ip.shape[0]-winsize+1):
                for j in range(grayscale_ip.shape[1]-winsize+1):
                    image_windows[idx] = (grayscale_ip[i:i+winsize,j:j+winsize].flatten())
                    idx += 1
            
            #Calculating ssd of output patch with respect to input patch, ensuring mask is applied so that similar neighbourhoods are considered.
            ssd = np.sum((((image_windows - padded_output_window)**2)*mask), axis=1)
            
            #Ids that match the error threshold over minimum ssd.
            ids = np.where(ssd <= 1.1*np.min(ssd))[0]
            
            #Picking random index.
            randi = np.random.choice(ids)

            #Calculating pixel row and column from flattened index.
            pixel_row = randi//(img.shape[1]-winsize+1)
            pixel_col = randi%(img.shape[1]-winsize+1)

            #Setting the value of output pixel
            output_image[row, column] = grayscale_ip[pixel_row+(winsize//2), pixel_col+(winsize//2)]
            binary_mask[row,column] = 1

            #Ensuring padded output is updated.
            padded_output_img[winsize//2:output_image.shape[0]+winsize//2, winsize//2:output_image.shape[1]+winsize//2] = output_image

    end = time.time()
    print("Win Size: ", winsize)     
    print("runTime: ", end - start)

    io.imshow(output_image, cmap = "gray")
    io.show()
       
    return output_image


# Non-parametric Texture Synthesis using Efros & Leung algorithm  
winsize = 3 # specify window size (5, 7, 11, 15)
outSize = 70 # specify size of the output image to be synthesized (square for simplicity)
# implement the following, save the synthesized image and record the run-times
im_synth = synthEfrosLeung(img, winsize, outSize)


#random patch
#random patch
def synthRandomPatch(im,tileSize,numTiles,outSize):
    start = time.time()
    im_height,im_width,num_channels = np.shape(im)
    #generate random tile
    tile_row = np.random.randint(0, im_height - tileSize + 1)
    tile_col = np.random.randint(0, im_width - tileSize + 1)
    random_tile = im[tile_row:tile_row + tileSize, tile_col:tile_col + tileSize]
    #repeat tile in left to right, top to bottom manner
    output_image = np.tile(random_tile, (numTiles, numTiles, 1))
    end = time.time()
    print(end-start)
    io.imshow(output_image, cmap = "gray")
    io.show()
    return output_image
# Random patches
tileSize = 30 # specify block sizes
numTiles = 5
outSize = numTiles * tileSize # calculate output image size
# implement the following, save the random-patch output and record run-times
#im_patch = synthRandomPatch(img, tileSize, numTiles, outSize)

