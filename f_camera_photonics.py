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
#import os
#os.chdir('C:\\folder_with_image_file')

varargin = 0

def f_camera_photonics(filename, varargin = 0):
#    filename = "smb_tminput_shortsp_shortrt_tmout.tiff"
    if varargin == 0:
        nports = 1
        box = []
        x_set = []
        y_set = []
        radius_set = []
    else:
        box = varargin[0]
        x_set = box[:,0]
        y_set = box[:,1]
        radius_set = box[:,3]
        nports = len(x_set)
    
    
    def make_pixels_black(img, dimensions, value=0):
        if (len(dimensions) == 4):
            for x in range(dimensions[1], dimensions[1]+dimensions[3]):
                for y in range(dimensions[0],dimensions[0]+dimensions[2]):
                    img[x,y] = value;
        return img        
            
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
    
    #open file as array of uint8 for veiwing and selecting
    img = cv2.imread(filename,0)
    img_array = np.array(img)
#    plt.imshow(img_array)
    #open file as full 16 bit tiff image
    img2 = cv2.imread(filename,-1)
    
    img_8bit = np.log(img)
    img_8bit = img_8bit - np.min(np.min(img_8bit))
    img_8bit = 255 - img_8bit / (np.max(np.max(img_8bit)))*255
    img_8bit = np.array(img_8bit, dtype = np.uint8)
    img_array = img_8bit
    
    
    ## WARNING: DO NOT ATTEMPT TO SHOW THE 16 bit file because it will probably crash python 
    print("\n\nSelect a region of pixels which you want to be blacked out. Every pixel inside this box will be set to black. When you see the white box hit enter or space.")
    r = cv2.selectROI(windowName="Black out Region Selector", img=img_array)
    img_array = make_pixels_black(img_array, r,255)
    img2 = make_pixels_black(img2, r)
    cv2.destroyWindow("Black out Region Selector")
    
    print("\n\nSelect a region of pixels which will be used to calculate the mean value of the image. This region should not contain the black region above, or any of the bright spots on the image. When you see the white box, hit enter or space.")
    mean_r = cv2.selectROI(windowName="Mean Pixel Value Selector", img=img_array)
    mean_value = calculate_pixel_mean(img_array, mean_r)
    mean_value2 = calculate_pixel_mean(img2, mean_r)
    
    #print(mean_value)
    #print(mean_value2)
    cv2.destroyWindow("Mean Pixel Value Selector")
    #cv2.imshow('image',img)
    
    #cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    #/////////////////////////////////////////////////////////////////////////////////////////////
    # End User Interface portion and begin calculations
    # Note: the pixel mean calculation was already performed in the function calculate_pixel_mean
    # Note: the black box was already created as well
    #/////////////////////////////////////////////////////////////////////////////////////////////
    
    #%%
    
    # Set Parameters
    radius = 3 #radius of spot size for each port
    row = 512
    col = 640 # expected image size
    #bright_pixel_threshold = 5000 #16 bit minimum value of a "bright" pixel
    
    min_residual = 1.03
    if(type(img2[0,0]) == np.uint16):
        saturation_level = math.pow(2,16)-3000
    elif(type(img2[0,0]) == np.uint8):
        saturation_level = math.pow(2,8)-30
    elif(type(img2[0,0]) == np.uint32):
        saturation_level = math.pow(2,32)-10000000
    pixel_increment = 3
    #bloom = 1
    rmax = 10
    
    img2 = np.subtract(img2, mean_value2) #subtract mean value from entire image
    maxval = np.max(img2) #find max value
    img2[img2 < 0] = 0 # set all negative values to 0
    
    if(maxval >= saturation_level):
        print("## ERROR ##")
        print("Image Saturated!")
        sys.exit()
    
    # Lines 57-63 of matlab file f_camera_photonics.m was ignored here
        
        
    # processing to find the gratings/ports
     # convolution-like operation scanning around the image to find power
    P_window = []
    figures = []
    
    if(len(box) <= 0): #if box exists
        for i in range(radius*2+1, row-radius*2-pixel_increment, pixel_increment): #step by pixel increment
            for j in range(radius*2+1, col-radius*2-pixel_increment, pixel_increment):
                subregion = img2[i-radius:i+radius, j-radius:j+radius]
                P = np.sum(subregion)
                P_window.append([P, i, j])
                
        P_window = np.array(P_window)
        M,I = img2.max(0),img2.argmax(0)
        #m = np.max(P_window)
        #ind = np.argmax(P_window)
        
        #More Parameters
        P_ports = np.array([[]])
        prev_x = []
        prev_y = []
        prev_radius = []
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
                if(abs(prev_x[i]-x) < prev_radius[i]*2) and (abs(prev_y[i] -y) < prev_radius[i]*2):
                    P_window[I[0],0] = 0 # null out anything close to the previous peak
                    near_pixel = 1

            
            if (near_pixel == 0):
                subregion = img2[int(x-radius):int(x+radius), int(y-radius):int(y+radius)]
                power_prev = np.sum(subregion)
                power_current = power_prev
                r = radius
                residual=2;
                while((residual > min_residual) and (r < rmax)):
                    r = r+1
                    subregion = img2[int(x-r):int(x+r), int(y-r):int(y+r)]                    
                    power_current = np.sum(subregion)
                    residual = power_current/power_prev
                    power_prev = power_current
                    
                plt.figure(fig_count)
                plt.imshow(np.array(subregion))
                plt.title("Spot at X= " + str(y) + ", Y= " + str(x)) ## somehow it seems x and y are switched here...
                plt.colorbar()
                plt.show()
                figures.append(fig_count)
                fig_count += 1

                
                prev_x.append(x)
                prev_y.append(y)
                prev_radius.append(r)
                
                if(P_ports.size > 0):
                    P_ports = np.concatenate([P_ports, P_window[int(I[0])].reshape(1,3)])
                else:
                    P_ports = P_window[int(I[0])].reshape(1,3)
            near_pixel = 0
        
        # now we have radius, x and y values, calculate the actaul powers
        P = []
        P_norm = []
        for i in range(0, nports):
            x = P_ports[i, 1]
            y = P_ports[i, 2]
            this_radius = prev_radius[i]
            #dim2 = np.array([y/col-1*this_radius/col, (row-x)/row-1*this_radius/row, 2*this_radius/col, 2*this_radius/row])
            #dim3 = np.array([y/col-0.75*radius/col, (row-x)/row-0.75*radius/row, 2*radius/col, 2*radius/row])
            subregion = img2[int(x)-this_radius:int(x)+this_radius, int(y)-this_radius:int(y)+this_radius]
            P.append(np.sum(subregion))
            P_norm.append(P[i]/P[0])
            
            P_ports[i,0] = P[i] 

        for i, x in enumerate(figures):
            #add power to figures
            plt.figure(x)
            plt.annotate("Normalized Power: " + str(P_norm[i]), (0,0.05), xycoords ='axes fraction', color="white")
            
        
        radius = prev_radius
    #    P_ports = np.sort(P_ports)
        x_vec=P_ports[:,1]
        y_vec=P_ports[:,2]
    else:
        x_vec = x_set
        y_vec = y_set
        radius = radius_set
        for i in range(0, nports):
            this_radius = radius[i]
            x = x_vec[i]
            y = y_vec[i]
            subregion = img2[x-this_radius:x+this_radius, y-this_radius:y+this_radius]
            P[i] = np.sum(subregion)
        P_ports = np.array([P.transpose(), x_vec, y_vec])


        
        
    for i in range(0, nports):
        cv2.rectangle(img_array, (int(P_ports[i,2]-radius[i]), int(P_ports[i,1]-radius[i])), (int(P_ports[i,2] + radius[i]), int(P_ports[i,1]+radius[i])), (0,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        annotation = "(" + str(P_ports[i,2]) + ", " + str(P_ports[i,1]) + ")"
        cv2.putText(img_array,annotation,(int(P_ports[i,2]-4*radius[i]),int(P_ports[i,1]-2*radius[i])), font, 0.3,(0,0,0),1,cv2.LINE_AA)
    cv2.imshow('Spot Locations Image',img_array)
#    plt.figure()
#    plt.imshow(img_array)
#    plt.title("Location of Selected Spots")

    
    pout = {"Total Power": P_ports[:,0], "Normalized Power":P_norm, "x":P_ports[:,1], "y":P_ports[:,2], "radius":radius}
#    print("\n\nPress 0 to close the image and return the function")
#    cv2.waitKey(0)
#    cv2.destroyWindow("Spot Locations Image")
    return pout
            
        
    
if(__name__ == "__main__"):
    filename = "smb_tminput_shortsp_shortrt_tmout.tiff"
    pout = f_camera_photonics(filename, varargin = 0)
    print(pout)
    
    
    
    
    
    
    
    
    
    #%%