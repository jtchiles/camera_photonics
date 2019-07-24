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

class objectview(object): pass # allows thing.x rather than thing['x']

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
    valid_box_str = Config.get("signal_processing","valid_box")
    cfg.valid_box=eval(valid_box_str)  # should give a 2x2 list or None
    cfg.box_width=Config.getint("signal_processing","box_width")
    cfg.max_box_width=Config.getint("signal_processing","max_box_width")
    cfg.kernel_width=Config.getint("signal_processing","kernel_width")
    cfg.min_residual=Config.getfloat("signal_processing","min_residual")
    cfg.min_standoff=Config.getfloat("signal_processing","min_standoff")
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


def pick_ports(image, nports, cfg=None):
    ''' The main peakfinding algorithm.
        convolution-like operation scanning around the image to find power.
    '''
    if cfg is None:
        cfg = get_all_config()

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
            net_standoff = max(prev_box_width[i], cfg.min_standoff)
            if(abs(prev_x[i]-x) < prev_box_width[i]) and (abs(prev_y[i] -y) < prev_box_width[i]):
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

    port_array = PortArray()
    for iPort in range(nports):
        port_array.add_port(x=prev_x[iPort], y=prev_y[iPort], w=prev_box_width[iPort])
    return port_array

    # x_vec=P_ports[:,1]
    # y_vec=P_ports[:,2]
    # box_width_vec = prev_box_width
    # return x_vec, y_vec, box_width_vec


class PortArray(object):
    ''' Handles sorting. No connection with image, but it can be fed an image to get powers at those points.
        You can get the port info by index, or you can get the vectors: x_vec, y_vec, w_vec, P_vec, Pnorm_vec.
        x, y, and w are stored as floats
    '''
    default_sortkey = 'position'

    def __init__(self, x_vec=None, y_vec=None, w_vec=None, P_vec=None):
        if any(vec is not None for vec in [x_vec, y_vec, w_vec]):
            if any(vec is None for vec in [x_vec, y_vec, w_vec]):
                raise ValueError('If any vector is specified to initialize PortArray, all of them must be specified')
            nports = len(x_vec)
            if any(len(vec) != nports for vec in [x_vec, y_vec, w_vec]):
                raise ValueError('x, y, and w vectors are not the same length')
            self.x_vec = np.array(x_vec)
            self.y_vec = np.array(y_vec)
            self.w_vec = np.array(w_vec)
        else:
            self.x_vec = np.array([])
            self.y_vec = np.array([])
            self.w_vec = np.array([])
        if P_vec is None:
            self._P_vec = None
        else:
            if len(P_vec) != len(x_vec):
                raise ValueError('Power vector a different length than the number of ports')
            self._P_vec = np.array(P_vec)

    def __len__(self):
        return len(self.x_vec)

    def __getitem__(self, index):
        ret_array = np.array([self.x_vec[index], self.y_vec[index], self.w_vec[index], 0])
        if self._P_vec is not None:
            ret_array[-1] = self.P_vec[index]
        return ret_array

    def add_port(self, x, y, w=15):
        ''' at the end, it also resorts this object by the default '''
        self.x_vec = np.append(self.x_vec, x)
        self.y_vec = np.append(self.y_vec, y)
        self.w_vec = np.append(self.w_vec, w)
        self.sort_by()

    @classmethod
    def from_boxspec(cls, box_spec):
        '''
            It is an iterable like [[x1, y1, width1] , [x2, y2, width2]],
            where "width" is side length of a rectangular box
        '''
        obj = cls()
        for port_spec in box_spec:
            x, y, w = port_spec
            obj.add_port(x, y, w)
        return obj

    def to_boxspec(self, with_powers=False):
        box_spec = []
        for iPort in range(len(self)):
            port_spec = self[iPort]
            if len(port_spec) == 4 and not with_powers:
                port_spec = port_spec[:3]
            box_spec.append(port_spec)
        return box_spec

    @property
    def P_vec(self):
        if self._P_vec is None:
            raise AttributeError('Power has not yet been measured')
        return self._P_vec

    @property
    def Pnorm_vec(self):
        if np.max(self.P_vec) == 0:
            print('Cannot normalize powers because they are all zero')
            return self.P_vec
        return self.P_vec / np.max(self.P_vec)

    def calc_powers(self, image, use_max=False):
        ''' Looks at the port positions within the given image.
            If use_max is False, the sum is used
        '''
        self._P_vec = np.zeros(len(self))
        for iPort in range(len(self)):
            x = int(self.x_vec[iPort])
            y = int(self.y_vec[iPort])
            w = int(self.w_vec[iPort])
            subregion = image[x-w:x+w, y-w:y+w]
            self._P_vec[iPort] = np.max(subregion) if use_max else np.sum(subregion)
        return self.P_vec

    def sort_by(self, keytype=None):
        '''
            keytype can be 'x', 'y', 'position', 'P'
            where 'position' is equivalent to 'x' or 'y' depending on which one varies the most.
        '''
        if keytype is None:
            keytype = self.default_sortkey
        if keytype == 'position':
            # Sort based on dimension of most position variance
            xvar=np.var(self.x_vec)
            yvar=np.var(self.y_vec)
            if xvar>yvar:
                sorting_vec = self.x_vec
            else:
                sorting_vec = self.y_vec
        elif keytype == 'x':
            sorting_vec = self.x_vec
        elif keytype == 'y':
            sorting_vec = self.y_vec
        elif keytype == 'P':
            sorting_vec = self.P_vec
        else:
            raise ValueError('Invalid sort key: {}. Must be in [\'x\', \'y\', \'position\', \'P\'].'.format(keytype))
        permutation = sorting_vec.argsort()
        self.x_vec = self.x_vec[permutation]
        self.y_vec = self.y_vec[permutation]
        self.w_vec = self.w_vec[permutation]
        if self._P_vec is not None:
            self._P_vec = self._P_vec[permutation]

    def to_dict(self):
        pout = {"Total Power": list(self.P_vec),
                "Normalized Power": list(self.Pnorm_vec),
                "x": list(self.x_vec),
                "y": list(self.y_vec),
                "box_width": list(self.w_vec)}
        return pout

    @classmethod
    def from_dict(cls, the_dict):
        obj = cls(the_dict['x'], the_dict['y'], the_dict['box_width'],
                  P_vec=the_dict.get('Total Power', None))
        return obj

    def __str__(self):
        return str(self.to_boxspec(with_powers=True))



#### Main function

def f_camera_photonics(filename, box_spec=None, configfile=None, **config_overrides):
    ''' Not backwards compatible! - filename is now relative to user's directory.

        To skip the user interface, give box_spec. It can be a PortArray or a list of lists.
    '''

    #first, get the os-independent directories and filenames put together
    filename = os.path.realpath(filename)
    filename_short = os.path.basename(filename)
    if os.path.splitext(filename)[1] == '.json':
        raise IOError('It looks like to are trying to apply image processing to a json file')

    cfg = get_all_config(configfile, **config_overrides)

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
        if cfg.valid_box is None:
            print("\n\nSelect a valid region of pixels to look for all output ports.  Everywhere else will be zeroed out.")
            windowName = filename_short + ' : Valid region selector'
            win = cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName, 800, 600)
            r = cv2.selectROI(windowName=windowName, img=img_array)
            print(r)
            cv2.destroyWindow(windowName)
        else:
            r = cfg.valid_box
        img_array = make_pixels_black(img_array, r,255)
        img2_temp=img2
        img2_temp=img2_temp*0
        img2 = swap_pixel_data(img2_temp, img2,r)


    #subtract darkfield (background) image from main data.
    if cfg.use_darkfield:
        img2=np.subtract(img2.astype(float),img_darkfield.astype(float))
        img2[img2 < 0] = 0 # set all negative values to 0
        print("The maximum value in the image after darkfield correction is: "+str(np.amax(img2)) +" (out of a camera limit of " +str(math.pow(2, cfg.bit_depth_per_pixel)-1)+")")

    saturation_level=math.pow(2,cfg.bit_depth_per_pixel)*cfg.saturation_level_fraction #calculate the threshold for the saturation condition
    maxval = np.amax(img2) #find max value in the entire image

    #check if saturation has occurred
    if(maxval >= saturation_level):
        raise RuntimeError("Image Saturated!")


    #/////////////////////////////////////////////////////////////////////////////////////////////
    # End User Interface portion and begin calculations
    # At this point, any desired background and ROI corrections have been completed.
    #/////////////////////////////////////////////////////////////////////////////////////////////

    # find the gratings/ports
    user_check_required = (box_spec is None) or True
    if box_spec is not None:
        if isinstance(box_spec, PortArray):
            port_arr = box_spec
        else:
            port_arr = PortArray.from_boxspec(box_spec)
        nports = len(port_arr)
    else:
        nports = cfg.default_nports
        port_arr = pick_ports(img2, nports, cfg)
    port_arr.calc_powers(img2)

    #draw a rectangle surrounding each port to represent the integration window.
    # This currently has big problems referring to old variables
    img2_scaled=img2/(math.pow(2,cfg.bit_depth_per_pixel)/16)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.3
    fontColor              = (1,1,1)
    lineType               = 1
    for i in range(0, nports):
        this_port = port_arr[i]
        cv2.rectangle(img2_scaled,
                      (int(this_port[1] - this_port[2]), int(this_port[0] - this_port[2])),
                      (int(this_port[1] + this_port[2]), int(this_port[0] + this_port[2])),
                      (0,255,0), 1)

        annotation = "(#"+ str(i+1) +", " + "P: " +"{:.2f}".format(port_arr.Pnorm_vec[i])+")"
        #if you want to include the row and column indices, tack this on to the annotation: +", " + str(P_ports[i,2]) + ", " + str(P_ports[i,1]) +
        print(annotation)

        #put the annotation near but slightly offset from the port location
        side_sign = int(2 * (i % 2 - .5))
        if i % 2 == 0:
            text_offset = this_port[2]
        else:
            text_offset = - this_port[2] - 6 * len(annotation)
        location = (int(this_port[1] + text_offset),
                    int(this_port[0] - 0.3 * this_port[2]))

        cv2.putText(img2_scaled,
            annotation,
            location,
            font,
            fontScale,
            fontColor,
            lineType)

    if user_check_required:
        print('Check if it is correct')
        windowName = filename + ' : Peakfinder results'
        cvshow(img2_scaled, windowName=windowName)

    pout = port_arr.to_dict()
    return pout


def save_output(some_dict, filename):
    ''' Takes any dictionary. Makes the file human readable.
        Converts np.ndarray to list first (disabled)
    '''
    # for k, dat in some_dict.items():
    #     if isinstance(dat, np.ndarray):
    #         some_dict[k] = list(dat)
    with open(filename, 'w') as fx:
        json.dump(some_dict, fx, sort_keys=True, indent=4)


def load_output(filename):
    ''' Converts all lists to np.ndarrays after loading (disabled) '''
    with open(filename, 'r') as fx:
        data_dict = json.load(fx)
    # for k, dat in data_dict.items():
    #     if isinstance(dat, (list, tuple)):
    #         data_dict[k] = np.array(dat)
    return data_dict


def main(filename, box_spec=None, **config_overrides):
    ''' Basically a wrapper for the f_camera_photonics algorithm, with saving '''
    pout = f_camera_photonics(filename, box_spec=box_spec, **config_overrides)

    directory, base = os.path.split(filename)
    json_basename = os.path.splitext(base)[0] + '.json'
    print('Saving to {} in {}'.format(json_basename, directory))
    save_output(pout, os.path.join(directory, json_basename))
    return pout


### Image file convenience processing ###
def cvshow(cvimg, windowName='img'):
    ''' Convenience of window sizing, cleanup, etc '''
    print('Press any key to close the display window')
    # big = cv2.resize(cvimg, (0,0), fx=3, fy=3)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 800, 600)
    cv2.imshow(windowName, cvimg)
    cv2.waitKey(0)
    cv2.destroyWindow(windowName)

def fix_tiff_range(filename=''):
    if os.path.isdir(filename):
        for file in iglob('*[(.tif)(.tiff)]'):
            fix_tiff_range(file)
    else:
        img = cv2.imread(filename, 0)
        new_filename = os.path.splitext(filename)[0] + '_proc.tif'
        cv2.imwrite(new_filename, 16 * img)


if(__name__ == "__main__"):
    if len(sys.argv) < 2:
        filenames = ["first_look-nolamp.tif"]
    else:
        filenames = sys.argv[1:]

    for fn in filenames:
        main(fn)


    #%%
