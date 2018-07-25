# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:32:57 2018

@author: jlm7
"""

from PIL import Image, ImageDraw
import numpy as np
import cv2
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.text
import configparser as cp
import os


camera_photonics_directory = os.path.dirname(os.path.realpath(__file__))

def get_all_config(configfile=None, **overrides):
    ''' Default is config.ini - relative to this file's directory.
        If specified, the file is relative to caller's working directory.
    '''
    if configfile is None:
        configfile = os.path.join(camera_photonics_directory, 'config.ini')
    configfile = os.path.realpath(configfile)

    Config = cp.ConfigParser()
    print(str(configfile))
    Config.read(configfile)

    class objectview(object): pass # allows thing.x rather than thing['x']

    #collect the config file's data, put it in an objectclass objectview(object):
    cfg = objectview()
    cfg.col=Config.getint("camera_attributes","horizontal_resolution")
    cfg.row=Config.getint("camera_attributes","vertical_resolution")
    cfg.bit_depth_per_pixel=Config.getint("camera_attributes","bit_depth_per_pixel")
    cfg.use_darkfield=Config.getboolean("camera_attributes","use_darkfield")
    cfg.use_brightfield=Config.getboolean("camera_attributes","use_brightfield")
    rel_darkfield_filename=Config.get("camera_attributes","darkfield_filename")
    cfg.darkfield_filename=os.path.join(camera_photonics_directory, rel_darkfield_filename)
    cfg.brightfield_filename=Config.get("camera_attributes","brightfield_filename")

    cfg.use_blanking_box=Config.getboolean("signal_processing","use_blanking_box")
    cfg.use_valid_box=Config.getboolean("signal_processing","use_valid_box")
    cfg.box_width=Config.getint("signal_processing","box_width")
    cfg.max_box_width=Config.getint("signal_processing","max_box_width")
    cfg.min_residual=Config.getfloat("signal_processing","min_residual")
    cfg.pixel_increment=Config.getint("signal_processing","pixel_increment")
    cfg.default_nports=Config.getint("signal_processing","default_nports")
    cfg.saturation_level_fraction=Config.getfloat("signal_processing","saturation_level_fraction")

    cfg.__dict__.update(overrides)

    return cfg


def f_camera_photonics(filename, varargin = 0):

    #first, get the os-independent directories and filenames put together
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    cfg = get_all_config()
    box_width = cfg.box_width  # temporary, because its used so much

    if varargin == 0:
        nports = cfg.default_nports
        box = []
        x_set = []
        y_set = []
        box_width_set = []
    else:
        box = varargin[0]
        x_set = box[:,0]
        y_set = box[:,1]
        box_width_set = box[:,3]
        nports = len(x_set)

    #null out pixels within an ROI
    def make_pixels_black(img, dimensions, value=0):
        if (len(dimensions) == 4):
            for x in range(dimensions[1], dimensions[1]+dimensions[3]):
                for y in range(dimensions[0],dimensions[0]+dimensions[2]):
                    img[x,y] = value
        return img

    #replace the pixels in an ROI of one array with those of another array
    def swap_pixel_data(img_orig, img_new, dimensions):
        img_temp=img_orig
        if (len(dimensions) == 4):
            for x in range(dimensions[1], dimensions[1]+dimensions[3]):
                for y in range(dimensions[0],dimensions[0]+dimensions[2]):
                    img_temp[x,y] = img_new[x,y]
        return img_temp

    #calculate mean value over an ROI
    def calculate_pixel_mean(img, dimensions):
        pixel_count = 0
        pixel_sum = 0
        if (len(dimensions) == 4):
            for x in range(dimensions[1], dimensions[1]+dimensions[3]):
                for y in range(dimensions[0],dimensions[0]+dimensions[2]):
                    pixel_count += 1
                    pixel_sum = pixel_sum + img[x,y]
        pixel_avg = pixel_sum / pixel_count
        return pixel_avg

    #open file as array of uint8 for viewing and selecting
    img = cv2.imread(filename,0)
    img_array = np.array(img)

    if(cfg.use_darkfield is True):
        img_darkfield=cv2.imread(cfg.darkfield_filename,-1)
        

    #open file as full 16 bit tiff image
    img2 = cv2.imread(filename,-1)
    max_img8bit = np.max(img)
    scaled_img = img*max_img8bit
    
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
#   IF THE IMAGE SHOWS ALL BLACK, UNCOMMENT THIS CODE AND COMMENT THE BELOW SECTION
    
    img_8bit = scaled_img - np.min(scaled_img)

    img_8bit = np.log10(img_8bit)
    img_8bit = 255 - img_8bit / (np.max(img_8bit))*255
    img_8bit = np.array(img_8bit, dtype = np.uint8)
    img_array = img_8bit

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # ONLY ONE OF THESE TWO SECTIONS SHOULD BE UNCOMMENTED AT A TIME

    # img_8bit = np.log(img)
    # img_8bit = img_8bit - np.min(img_8bit)
    # img_8bit = 255 - img_8bit / (np.max(img_8bit))*255
    # img_8bit = np.array(img_8bit, dtype = np.uint8)
    # img_array = img_8bit
    # plt.imshow(img_array)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #If the user chooses to use a "blanking box", open an ROI selector and null out everything inside the ROI.
    if cfg.use_blanking_box is True:
        print("\n\nSelect a region of pixels which you want to be zeroed out. Every pixel inside this box will be set to black. When you see the white box hit enter or space.")
        r = cv2.selectROI(windowName="Black out Region Selector", img=img_array)
        img_array = make_pixels_black(img_array, r,255)
        img2 = make_pixels_black(img2, r)
        cv2.destroyWindow("Black out region selector")

    #If the user chooses to use a "valid box", open an ROI selector and null out everything outside the ROI.
    if cfg.use_valid_box is True:
        print("\n\nSelect a valid region of pixels to look for all output ports.  Everywhere else will be zeroed out.")
        r = cv2.selectROI(windowName="Valid region selector", img=img_array)
        img_array = make_pixels_black(img_array, r,255)
        img2_temp=img2
        img2_temp=img2_temp*0
        img2 = swap_pixel_data(img2_temp, img2,r)
        cv2.destroyWindow("Valid region selector")


    #subtract darkfield (background) image from main data.
    img2=np.subtract(img2.astype(float),img_darkfield.astype(float))   
    img2[img2 < 0] = 0 # set all negative values to 0
    print("The maximum value in the image after darkfield correction is: "+str(np.amax(img2)) +" (out of a camera limit of " +str(math.pow(2, cfg.bit_depth_per_pixel)-1)+")")


    #/////////////////////////////////////////////////////////////////////////////////////////////
    # End User Interface portion and begin calculations
    # At this point, any desired background and ROI corrections have been completed.
    #/////////////////////////////////////////////////////////////////////////////////////////////

    
    saturation_level=math.pow(2,cfg.bit_depth_per_pixel)*cfg.saturation_level_fraction #calculate the threshold for the saturation condition
    maxval = np.amax(img2) #find max value in the entire image

    #check if saturation has occurred
    if(maxval >= saturation_level):
        print("## ERROR ##")
        print("Image Saturated!")
        sys.exit()

    # processing to find the gratings/ports
     # convolution-like operation scanning around the image to find power
    P_window = []
    figures = []

    if(len(box) <= 0): #if box exists
        for i in range(box_width*2+1, cfg.row-box_width*2-cfg.pixel_increment, cfg.pixel_increment): #step by pixel increment
            for j in range(box_width*2+1, cfg.col-box_width*2-cfg.pixel_increment, cfg.pixel_increment):
                subregion = img2[i-box_width:i+box_width, j-box_width:j+box_width]
                P = np.sum(subregion)
                P_window.append([P, i, j])

        P_window = np.array(P_window)
        M,I = img2.max(0),img2.argmax(0)

        #More Parameters
        P_ports = np.array([[]])
        prev_x = []
        prev_y = []
        prev_box_width = []
        near_pixel = 0
        fig_count = 1

        #find the top nports candidates based on power
        while P_ports.shape[0] < nports or P_ports.shape[1] == 0:
            M,I = P_window.max(0),P_window.argmax(0)
            I2 = [P_window[I[0],1],P_window[I[0],2]]
            x=int(I2[0])
            y=int(I2[1])

            for i in range(0, len(prev_x)):
                #TODO fix line 112
                #see if the current candidate is too close to the last peak location
                if(abs(prev_x[i]-x) < prev_box_width[i]*3) and (abs(prev_y[i] -y) < prev_box_width[i]*3):
                    P_window[I[0],0] = 0 # null out anything close to the previous peak
                    near_pixel = 1


            if (near_pixel == 0):
                subregion = img2[int(x-box_width):int(x+box_width), int(y-box_width):int(y+box_width)]
                power_prev = np.sum(subregion)
                power_current = power_prev
                r = box_width
                residual=2
                while((residual > cfg.min_residual) and (r < cfg.max_box_width)):
                    r = r+1
                    subregion = img2[int(x-r):int(x+r), int(y-r):int(y+r)]
                    power_current = np.sum(subregion)
                    if power_prev!=0:
                        residual = power_current/power_prev
                    else:
                        residual=0
                    power_prev = power_current

                fig_count += 1


                prev_x.append(x)
                prev_y.append(y)
                prev_box_width.append(r)

                if(P_ports.size > 0):
                    P_ports = np.concatenate([P_ports, P_window[int(I[0])].reshape(1,3)])
                else:
                    P_ports = P_window[int(I[0])].reshape(1,3)
            near_pixel = 0

        # now we have box_width, x and y values, calculate the actual powers from the real data
        P = []
        P_norm = []
        for i in range(0, nports):
            x = P_ports[i, 1]
            y = P_ports[i, 2]
            this_box_width = prev_box_width[i]
            subregion = img2[int(x)-this_box_width:int(x)+this_box_width, int(y)-this_box_width:int(y)+this_box_width]
            P.append(np.sum(subregion))
            P_norm.append(P[i])

            P_ports[i,0] = P[i]
        box_width = prev_box_width
        x_vec=P_ports[:,1]
        y_vec=P_ports[:,2]
    else:
        x_vec = x_set
        y_vec = y_set
        box_width = box_width_set
        for i in range(0, nports):
            this_box_width = box_width[i]
            x = x_vec[i]
            y = y_vec[i]
            subregion = img2[x-this_box_width:x+this_box_width, y-this_box_width:y+this_box_width]
            P[i] = np.sum(subregion)
        P_ports = np.array([P.transpose(), x_vec, y_vec])

    xvar=np.var(P_ports[:,1])
    yvar=np.var(P_ports[:,2])
    if xvar>yvar:
        P_ports=P_ports[P_ports[:,1].argsort()]
        print("Detected that the ports should be sorted along the row index.  Proceeding with this assumption.")
    else:
        P_ports=P_ports[P_ports[:,2].argsort()]
        print("Detected that the ports should be sorted along the column index.  Proceeding with this assumption.")
    P_norm=P_ports[:,0]/np.amax(P_ports[:,0])


    img2_scaled=img2/(math.pow(2,cfg.bit_depth_per_pixel)/16)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    
    fontScale              = 0.3
    fontColor              = (1,1,1)
    lineType               = 1
    for i in range(0, nports): 
        #draw a rectangle surrounding each port to represent the integration window.
        cv2.rectangle(img2_scaled, (int(P_ports[i,2]-box_width[i]), int(P_ports[i,1]-box_width[i])), (int(P_ports[i,2] + box_width[i]), int(P_ports[i,1]+box_width[i])), (32,32,32), 1)
        
        annotation = "(#"+ str(i+1) +", " + "P: " +"{:.2f}".format(P_norm[i])+")"
        #if you want to include the row and column indices, tack this on to the annotation: +", " + str(P_ports[i,2]) + ", " + str(P_ports[i,1]) + 
        print(annotation)

        #put the annotation near but slightly offset from the port location
        location = (int(P_ports[i,2]+box_width[i]),int(P_ports[i,1]-0.3*box_width[i]))

        cv2.putText(img2_scaled,
            annotation, 
            location, 
            font, 
            fontScale,
            fontColor,
            lineType)
    #scale it up for readability
    big=cv2.resize(img2_scaled, (0,0), fx=3, fy=3) 
    cv2.imshow("img2",big)


    #cv2.waitKey(0)

    
    pout = {"Total Power": P_ports[:,0], "Normalized Power":P_norm, "x":P_ports[:,1], "y":P_ports[:,2], "box_width":box_width}
    print("\n\nPress 0 to close the image and return the function")
    cv2.waitKey(0)
    cv2.destroyWindow("img2")
    return pout



if(__name__ == "__main__"):
#    filename = "grating1_TE_1310nm_300.tiff"
    filename = "first_look-nolamp.tif"

    pout = f_camera_photonics(filename, varargin = 0)
    print(pout)









    #%%
