import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt

class SimulatedEnvironment:

    def create_image(self, filename=""):
        # Create a black image and draw box
        '''img = np.zeros((200,200,3), np.uint8)
        cv2.rectangle(img,(100,20),(180,180),(0,0,255),5)
        # Hard coded rec
        cv2.rectangle(img,(124,8),(223,223),(0,0,255),2)'''
        img = cv2.imread(filename)
        #ret,img = cv2.threshold(img,10,200,cv2.THRESH_BINARY)
        return img

    def move(self, x, y, img):
        copy = img.copy()
        new_img = cv2.circle(copy,(x,y),8,(255,0,0),2)
        return new_img

    def create_light(self,img):
        light_img = img.copy()
        peaks = [(9,216), (22,216), (35,214), (48,214), (63,211), (75,211), (89,208), (103, 208), (116,205), (129,205), (143, 202), (156, 202), (169, 199), (183, 198), (197, 194), (210, 194)]
        for i in range(len(peaks)):
            y, x = peaks[i]
            light_img = cv2.circle(light_img,(x,y),2,(0,255,255),2)
        return light_img

    def intensity(self, peaks, fiber_pos, b):
        x, y = fiber_pos
        x0 = 107
        y0 = 96
        gamma = 10
        u, v = np.meshgrid(np.arange(gamma), np.arange(gamma))
        r = math.sqrt((x-x0)**2 +(y-y0)**2)
        A = math.exp((-r/(2*gamma))**2)
        z = b * A
        q = []
        q_gaus = []
        # for all peaks
        for i in range(len(peaks)):
            u0, v0 = peaks[i]
            q.append( np.sqrt((u-u0)**2 +(v-v0)**2))
            q_gaus.append( math.exp((-q[i]/(2*gamma))**2))
        p = z* sum(q_gaus)
        return p

    def prob_1 (self, n, peaks):
        # Whats the (x,y) that gives max I at peak n?
        b = 1
        window = 20
        # Start w/ random guess
        x = 100
        y = 100
        I0 = []
        # Get x and y that gives max I for peak 0
        for i in range(-window/2, window/2):
            x+= i
            y = 100
            for j in range (-window/2, window/2):
                y+= j
                I = self.intensity(peaks, (x, y), b)
                # Store as tuple: (Intensity, (x,y))
                I0.append((I[n], (x,y))
        return max(I0)[1]

    def prob_2(self, attin, peaks):
        I = []
        for i in range(1):
            #range(len(attin))
            #img = cv2.imread(imgs[i])
            b = attin[i]
            # peaks stored as tuple
            I.append(self.intensity(peaks, (107, 96), b))
        return I

    def is_pos_correct(self, x, y):
         if (x in range(93,99)) and (y in range(104,110)):
             return True
         return False

    def key_control(self, filename=""):
        img = self.create_image(filename)
        y = 93
        x = 15
        new_img = self.move(x, y, img)
        # Hard coded key strokes (WASD)
        while True:
            print(x, y)
            if self.is_pos_correct(x, y):
                new_img = self.move(x, y, img)
                new_img = self.create_light(new_img)
            #else new_img = self.move(x, y, img)
            cv2.imshow('image', new_img)
            key = cv2.waitKeyEx()
            # Up/W = 119
            if (key == 119):
                y-=1
                new_img = self.move(x, y, img)
            # Down/S = 115
            if (key == 115):
                y+=1
                new_img = self.move(x, y, img)
            # Left/A = 97
            if (key == 97):
                x-=1
                new_img = self.move(x, y, img)
            # Right/D = 100
            if (key == 100):
                x+=1
                new_img = self.move(x, y, img)
            # Space to exit
            if (key == 32):
                break


#SUWG01_01-lamp.tif

sim = SimulatedEnvironment()
#imgs = ['SUWG01_01-20dB.tif','SUWG01_01-13dB.tif','SUWG01_01-10dB.tif', 'SUWG01_01-0dB.tif']
attin = [0.01, 0.05, 0.1, 1]
peaks = [(9,216), (22,216), (35,214), (48,214), (63,211), (75,211), (89,208), (103, 208), (116,205), (129,205), (143, 202), (156, 202), (169, 199), (183, 198), (197, 194), (210,194)]

#sim.prob_1(0, peaks)
sim.prob_2(attin, peaks)

#if len(sys.argv) > 1:
#    sim.key_control(str(sys.argv[1]))


