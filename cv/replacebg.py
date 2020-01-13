import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

#RESIZE IMAGE BY WIDTH OR HEIGHT MAINTAINING ASPECT RATIO
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

input_dir = 'input/'
output_dir = 'output/'
mask_dir = 'masks/'
bg_dir = 'bg/'

bg_file = 'track.jpg'
bg_name = bg_dir + bg_file

#TODO: RUN IN LOOP
fg_file = '000000000.jpg'

fg_name = input_dir + fg_file
fg_file_no_ext = os.path.splitext(fg_file)[0]
mask_name = mask_dir + fg_file_no_ext + '.png'
output_name = output_dir + fg_file_no_ext    + '.png'

bg = cv2.imread(bg_name)
fg = cv2.imread(fg_name)
mask = cv2.imread(mask_name)
mask2 = mask.copy()
fg2 = fg.copy()
bg2 = bg.copy()

#print('Mask Dimensions : ',mask2.shape)
#print('FG Dimensions : ',fg2.shape)
#print('BG Dimensions : ',bg.shape)

#openCV places height before width in shape[] array

bgWidth = bg2.shape[1]
bgHeight = bg2.shape[0]

fgWidth = fg2.shape[1]
fgHeight = fg2.shape[0]

#if fgWidth > fgHeight:
  #fg2 = maintain_aspect_ratio_resize(fg2, height=bgHeight)
#else:
#  fg2 = maintain_aspect_ratio_resize(bg2, width=bgWidth)

#resize foreground image proportionally to match background width (should be dynamic to width / height based on ratio)
fg2 = maintain_aspect_ratio_resize(fg2, width=bgWidth)
#fg2 = maintain_aspect_ratio_resize(fg2, height=bgHeight)

fgWidth = fg2.shape[1]
fgHeight = fg2.shape[0]

mask2 = maintain_aspect_ratio_resize(mask2, width=fgWidth)

#Blur mask
th, alpha = cv2.threshold(np.array(mask2),0,255, cv2.THRESH_BINARY)

#the params for blur ksize must be odd numbers
alpha = cv2.GaussianBlur(alpha, (21,21),0)

#above a certain rgb value, use the pixels from fg, otherwise use pixels from bg
bg2[np.where(alpha > 200)] = fg2[np.where(alpha > 200)]
output = bg2.copy()


#SHOW IMAGE
cv2.imshow('result.png', output)
cv2.waitKey()
cv2.destroyAllWindows()

#WRITE IMAGE
#cv2.imwrite( output_name, output );
print('FINISH: ' + output_name)