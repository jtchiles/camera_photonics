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
import json
from glob import iglob

camera_photonics_directory = os.path.dirname(os.path.realpath(__file__))

def get_all_config(configfile=None, **config_overrides):
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
    cfg.kernel_width=Config.getint("signal_processing","kernel_width")
    cfg.min_residual=Config.getfloat("signal_processing","min_residual")
    cfg.pixel_increment=Config.getint("signal_processing","pixel_increment")
    cfg.default_nports=Config.getint("signal_processing","default_nports")
    cfg.saturation_level_fraction=Config.getfloat("signal_processing","saturation_level_fraction")

    cfg.__dict__.update(config_overrides)

    return cfg


#### Pixel iterating modifier functions:

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


def pick_ports(image, nports, cfg):
    # convolution-like operation scanning around the image to find power
    P_window = []
    figures = []
    window_centers = lambda pixels: range(cfg.kernel_width*2+1,
                                          pixels-cfg.kernel_width*2-cfg.pixel_increment,
                                          cfg.pixel_increment)
    for i in (window_centers(cfg.row)):
        for j in (window_centers(cfg.col)):
            subregion = image[i-cfg.kernel_width:i+cfg.kernel_width, j-cfg.kernel_width:j+cfg.kernel_width]
            P = np.sum(subregion)
            P_window.append([P, i, j])

    P_window = np.array(P_window)
    M,I = image.max(0),image.argmax(0)

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
            r = cfg.box_width
            subregion = image[int(x-r):int(x+r), int(y-r):int(y+r)]
            power_prev = np.sum(subregion)
            power_current = power_prev
            residual=2
            while((residual > cfg.min_residual) and (r < cfg.max_box_width)):
                r = r+1
                subregion = image[int(x-r):int(x+r), int(y-r):int(y+r)]
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

    x_vec=P_ports[:,1]
    y_vec=P_ports[:,2]
    box_width_vec = prev_box_width
    return x_vec, y_vec, box_width_vec


#### Main function

def f_camera_photonics(filename, box_spec=None, configfile=None, **config_overrides):
    ''' Not backwards compatible! - filename is now relative to user's directory.

        To skip the user interface, give box_spec.
        It is an np.ndarray like [[x1, y1, width1] , [x2, y2, width2]], where "width" is side length of sqaure box
    '''

    #first, get the os-independent directories and filenames put together
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    filename = os.path.realpath(filename)
    filename_short = os.path.basename(filename)

    cfg = get_all_config(configfile, **config_overrides)

    if box_spec is None:
        nports = cfg.default_nports
    else:
        x_vec = box_spec[:,0]
        y_vec = box_spec[:,1]
        box_width_vec = box_spec[:,2]
        nports = len(x_vec)

    #open file as array of uint8 for viewing and selecting
    img = cv2.imread(filename,0)
    img_array = np.array(img)

    #open file as full 16 bit tiff image
    img2 = cv2.imread(filename,-1)
    max_img8bit = np.max(img)
    scaled_img = img*max_img8bit

    if(cfg.use_darkfield is True):
        img_darkfield=cv2.imread(cfg.darkfield_filename,-1)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #   IF THE IMAGE SHOWS ALL BLACK, SET THIS TO TRUE
    if True:
        img_8bit = scaled_img - np.min(scaled_img)

        img_8bit = np.log10(img_8bit)
        img_8bit = 255 - img_8bit / (np.max(img_8bit))*255
        img_8bit = np.array(img_8bit, dtype = np.uint8)
        img_array = img_8bit
    else:
        img_8bit = np.log(img)
        img_8bit = img_8bit - np.min(img_8bit)
        img_8bit = 255 - img_8bit / (np.max(img_8bit))*255
        img_8bit = np.array(img_8bit, dtype = np.uint8)
        img_array = img_8bit
        plt.imshow(img_array)

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
        windowName = filename_short + ' : Valid region selector'
        win = cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 800, 600)
        r = cv2.selectROI(windowName=windowName, img=img_array)
        img_array = make_pixels_black(img_array, r,255)
        img2_temp=img2
        img2_temp=img2_temp*0
        img2 = swap_pixel_data(img2_temp, img2,r)
        cv2.destroyWindow(windowName)


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
        raise RuntimeError("Image Saturated!")

    # find the gratings/ports
    if box_spec is None:
        n_ports = cfg.default_nports
        x_vec, y_vec, box_width_vec = pick_ports(img2, n_ports, cfg)
    else:
        x_vec, y_vec, box_width_vec = box_spec.T
        n_ports = len(x_vec)

    # calculate their powers
    P_vec = []
    for i in range(0, nports):
        x = int(x_vec[i])
        y = int(y_vec[i])
        w = box_width_vec[i]
        subregion = img2[x-w:x+w, y-w:y+w]
        P_vec.append(np.sum(subregion))
    P_ports = np.array([P_vec, x_vec, y_vec, box_width_vec]).T

    # Sort based on dimension of most position variance
    xvar=np.var(P_ports[:,1])
    yvar=np.var(P_ports[:,2])
    if xvar>yvar:
        P_ports=P_ports[P_ports[:,1].argsort()]
        print("Detected that the ports should be sorted along the row index. "
              "Proceeding with this assumption.")
    else:
        P_ports=P_ports[P_ports[:,2].argsort()]
        print("Detected that the ports should be sorted along the column index. "
              "Proceeding with this assumption.")
    P_norm=P_ports[:,0]/np.amax(P_ports[:,0])


    img2_scaled=img2/(math.pow(2,cfg.bit_depth_per_pixel)/16)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.3
    fontColor              = (1,1,1)
    lineType               = 1
    for i in range(0, nports):
        #draw a rectangle surrounding each port to represent the integration window.
        cv2.rectangle(img2_scaled,
                      (int(P_ports[i,2]-P_ports[i,3]), int(P_ports[i,1]-P_ports[i,3])),
                      (int(P_ports[i,2] + P_ports[i,3]), int(P_ports[i,1]+P_ports[i,3])),
                      (32,32,32), 1)

        annotation = "(#"+ str(i+1) +", " + "P: " +"{:.2f}".format(P_norm[i])+")"
        #if you want to include the row and column indices, tack this on to the annotation: +", " + str(P_ports[i,2]) + ", " + str(P_ports[i,1]) +
        print(annotation)

        #put the annotation near but slightly offset from the port location
        location = (int(P_ports[i,2]+P_ports[i,3]),int(P_ports[i,1]-0.3*P_ports[i,3]))

        cv2.putText(img2_scaled,
            annotation,
            location,
            font,
            fontScale,
            fontColor,
            lineType)
    #scale it up for readability
    windowName = filename_short + ' : Peakfinder results'
    big=cv2.resize(img2_scaled, (0,0), fx=3, fy=3)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 800, 600)
    cv2.imshow(windowName, big)


    #cv2.waitKey(0)

    out_data = P_ports.T.tolist()
    pout = {"Total Power": out_data[0],
            "Normalized Power":P_norm.tolist(),
            "x":out_data[1],
            "y":out_data[2],
            "box_width":out_data[3]}
    print("\n\nPress 0 to close the image and return the function")
    cv2.waitKey(0)
    # if not cfg.use_valid_box:
    cv2.destroyWindow(windowName)
    return pout


def save_output(some_dict, filename):
    ''' Takes any dictionary. Makes the file human readable. '''
    with open(filename, 'w') as fx:
        json.dump(some_dict, fx, sort_keys=True, indent=4)


def load_output(filename):
    with open(filename, 'r') as fx:
        return json.load(fx)


def main(filename, box_spec=None, **config_overrides):
    ''' Basically a wrapper for the f_camera_photonics algorithm, with saving '''
    pout = f_camera_photonics(filename, box_spec=box_spec, **config_overrides)

    directory, base = os.path.split(filename)
    json_basename = os.path.splitext(base)[0] + '.json'
    print('Saving to {} in {}'.format(json_basename, directory))
    save_output(pout, os.path.join(directory, json_basename))
    return pout


### Batch processing on directories ###

def process_directory(dirname='', box_spec=None, glob='*[(.tif)(.tiff)]'):
    pathpattern = os.path.join(dirname, glob)
    for fn in iglob(pathpattern):
        filebase = os.path.splitext(os.path.basename(fn))[0]
        pout = main(fn, box_spec)
    consolidate_data(dirname)


def consolidate_data(dirname=''):
    all_pout = dict()
    pathpattern = os.path.join(dirname, '*.json')
    os.remove(os.path.join(dirname, 'all_data.json'))
    for fn in iglob(pathpattern):
        filebase = os.path.splitext(os.path.basename(fn))[0]
        pout = load_output(fn)
        all_pout[filebase] = pout
    save_output(all_pout, os.path.join(dirname, 'all_data.json'))


### File naming convention ###

def index_to_name(index_dict):
    ''' Converts dict(x=3, y=5) to "x3y5".
        Keys will be sorted alphanumerically
    '''
    name = ''
    for dim_name in sorted(index_dict.keys()):
        name += dim_name
        ind = index_dict[dim_name]
        if type(ind) is not int or ind > 9 or ind < 1:
            raise ValueError('Bad index: {}. Only integers 1-9 are supported.'.format(ind))
        name += str(ind)
    return name


def name_to_index(name):
    ''' Converts "x3y5.tif" or "x3y5" to dict(x=3, y=5)

        raises ValueError if the naming convention does not work.

        Limited processor as of now: supports single digit only
    '''
    index_dict = dict()
    index_variables = list('wxyz')
    name = name.split('.')[0]  # remove suffix

    # Check naming convention
    conforming = True
    for char in name:
        if not char.isdigit() and not char in index_variables:
            conforming = False
    if not name[-1].isdigit():
        conforming = False
    if not conforming:
        raise ValueError('Name "{}" does not conform to the naming convention: "w1x3y2".'.format(name))

    # Parse the values
    for ivar in index_variables:
        try:
            pos = name.index(ivar)
        except ValueError:
            continue
        ival = name[pos + 1]
        index_dict[ivar] = int(ival)
    return index_dict


### Data analysis over multiple files ###

def calc_shape(data_dict):
    ''' Turn the dictionary of data entries into grid-like information,
        such as ((4, 5), ['x', 'y']) for a 4x5 grid calling its parameters "x" and "y".

        data_dict is what you would get from ``load_output('all_data.json')``
    '''
    dim_sizes = None
    for data_name, data_entry in data_dict.items():
        index_dict = name_to_index(data_name)

        if dim_sizes is None:
            dim_sizes = dict((k, 0) for k in index_dict.keys())
        elif set(index_dict.keys()) != set(dim_sizes.keys()):
            raise ValueError('Inconsistent number or name of parameter dimensions in {}'.format(data_name))

        for dim_name in dim_sizes.keys():
            dim_sizes[dim_name] = max(dim_sizes[dim_name], index_dict[dim_name])
    dim_names = sorted(dim_sizes.keys())
    shape = tuple(dim_sizes[dim] for dim in dim_names)
    return shape, dim_names


def default_entry_to_scalar(data_entry):
    return data_entry['Normalized Power'][1]


def convert_to_array(data_dict, entry_to_scalar=default_entry_to_scalar):
    ''' Populate an array with scalars based on a dictionary of data.
        data_dict is what you would get from ``load_output('all_data.json')``

        The shape of the array is determined by the grid signified by the data names.

        The scalars are derived using the ``entry_to_scalar`` argument,
        which is a function that changes often based on the type of device.
        Missing elements are NaN.
    '''
    shape, dim_names = calc_shape(data_dict)
    array_vals = np.empty(shape)
    array_vals[:] = np.nan
    for data_name, data_entry in data_dict.items():
        index_dict = name_to_index(data_name)
        index = tuple(index_dict[dim] - 1 for dim in dim_names)
        value = entry_to_scalar(data_entry)
        array_vals[index] = value
    return array_vals


def plot_1d(data_array):
    plt.plot(data_array, '-.')


def plot_2d(data_array):
    ''' Wrapper for ``pcolor`` with options that are good for gridded integer domains
    '''
    fi, ax = plt.subplots()
    colorvals = ax.pcolor(data_array.T)
    # Put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data_array.shape[0]) + 0.5)
    ax.set_yticks(np.arange(data_array.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(data_array.shape[0]) + 1)
    ax.set_yticklabels(np.arange(data_array.shape[1]) + 1)
    # Label and color
    cbar = plt.colorbar(colorvals)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.show()


# todo: compare power data against something else in order to plot

if(__name__ == "__main__"):
    if len(sys.argv) < 2:
        filenames = ["first_look-nolamp.tif"]
    else:
        filenames = sys.argv[1:]

    for fn in filenames:
        main(fn)


    #%%
