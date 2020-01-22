import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import keyboard
import sys

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
bg_file = 'space.jpg'
bg_name = bg_dir + bg_file

initial_threshold_val = 150
threshold_val = initial_threshold_val

#TODO - make dynamic based on num files in input directory
num_inputs = 40

output_image = None

def generate_image(iterator,doShow):
    iteration = None
    if iterator < 10:
        iteration = '0' + str(iterator)
    else:
        iteration = str(iterator)

    fg_file = '0000000' + iteration + '.jpg'

    fg_name = input_dir + fg_file
    fg_file_no_ext = os.path.splitext(fg_file)[0]
    mask_name = mask_dir + fg_file_no_ext + '.jpg'
    output_name = output_dir + fg_file_no_ext    + '.jpg'

    bg = cv2.imread(bg_name)
    fg = cv2.imread(fg_name)
    mask = cv2.imread(mask_name)
    mask2 = mask.copy()
    fg2 = fg.copy()
    bg2 = bg.copy()

    # print('Mask : ',mask2.shape)
    # print('FG : ',fg2.shape)
    # print('BG : ',bg.shape)

    #openCV places height before width in shape[] array

    bgWidth = bg2.shape[1]
    bgHeight = bg2.shape[0]

    fgWidth = fg2.shape[1]
    fgHeight = fg2.shape[0]

    fg2 = maintain_aspect_ratio_resize(fg2, width=bgWidth)
    mask2 = maintain_aspect_ratio_resize(mask2, width=bgWidth)

    fgWidth = fg2.shape[1]
    fgHeight = fg2.shape[0]

    if fgHeight > bgHeight:
        fg2 = maintain_aspect_ratio_resize(fg2, height=bgHeight)
        mask2 = maintain_aspect_ratio_resize(mask2, height=bgHeight)

    #Blur mask
    #the params for blur ksize must be odd numbers
    alpha = cv2.GaussianBlur(mask2, (21,21),0)

    #above a certain rgb value, use the pixels from fg, otherwise use pixels from bg
    
    # print("MASK:",alpha.shape)
    # print("FG: ",fg2.shape)
    # print("BG:",bg2.shape)
    global threshold_val
    
    bg2[np.where(alpha > threshold_val)] = fg2[np.where(alpha > threshold_val)]
    output = bg2.copy()

    if doShow:
        #SHOW IMAGE
        output_image = cv2.imshow('result.jpg', output)
        doUpdate = False
        while True:
            # The function waitKey waits for a key event infinitely (when delay<=0)
            k = chr(cv2.waitKey(-1)) 
            if k == 'x':                       # toggle current image
                threshold_val-=1
                print("threshold:",threshold_val)
                doUpdate = True
                break
            elif k == 'z':
                threshold_val+=1
                print("threshold:",threshold_val)
                doUpdate = True
                break
            elif k == 'i':
                iterator+=1
                print("new image",iterator)
                threshold_val = initial_threshold_val
                doUpdate = True
                break
            elif k == 's':
                print("write image")
                cv2.imwrite( output_name, output )
                break
            elif k == 'q':
                break

        cv2.destroyAllWindows()
        if doUpdate:
            generate_image(iterator,True)
        
    else:
        #WRITE IMAGE
        cv2.imwrite( output_name, output )
        print('FINISH: ' + output_name)

def generate_all_images(number_to_generate):
    for i in range(number_to_generate):
        generate_image(i,False)

def generate_images(start_index,end_index):
    for i in range(start_index, end_index):
        generate_image(i,False)

arguments = len(sys.argv) - 1

if arguments > 0:
    if sys.argv[1] == '--image':
        generate_image(int(sys.argv[2]),True)
    elif sys.argv[1] == '--all':
        #requires currently that you manually adjust num_inputs
        generate_all_images(num_inputs)
    elif sys.argv[1] == '--generate':
        generate_all_images(int(sys.argv[2]))
    elif sys.argv[1] == '--start':
        if arguments > 2:
            if sys.argv[3] == '--end':
                generate_images(int(sys.argv[2]),int(sys.argv[4]))
            else: 
                generate_images(int(sys.argv[2]),num_inputs)
        else:
            generate_images(int(sys.argv[2]),num_inputs)
else:
    generate_image(0,True)