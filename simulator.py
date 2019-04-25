import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt

class SimulatedEnvironment:
    ## Finds peaks ##
    '''To do: only find peaks by threshold and move 10 pixels away'''
    def make_pixels_black(img, dimensions, value=0):
        if (len(dimensions) == 4):
            for x in range(dimensions[1], dimensions[1]+dimensions[3]):
                for y in range(dimensions[0],dimensions[0]+dimensions[2]):
                    img[x,y] = value
        return img

    def swap_pixel_data(img_orig, img_new, dimensions):
        img_temp=img_orig
        if (len(dimensions) == 4):
            for x in range(dimensions[1], dimensions[1]+dimensions[3]):
                for y in range(dimensions[0],dimensions[0]+dimensions[2]):
                    img_temp[x,y] = img_new[x,y]
        return img_temp

    def get_max_locs(img, row, col, max_value):
        max_locs = []
        for i in range(row):
            for j in range(col):
                if(img[i,j] == max_value):
                    max_locs.append((j, i))
        return max_locs

    def manual_box_finder(img, pos_first, pos_last, window):
        # get a least-squares solution to align peaks
        points = [pos_first,pos_last]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords,np.ones(len(x_coords))]).T
        m, b = np.linalg.lstsq(A, y_coords)[0]
        y, x = pos_first
        for i in range(16):
            cv2.circle(img, (y, x), window, (255, 0, 0), 1)
            y = int((x-b)/m)
            x += int(13)
        return img

    def set_peaks(file_lamp, file_dark):
        #open images
        img = cv2.imread(file_lamp, 0)
        img2 = cv2.imread(file_dark,0)
        img_array = np.array(img2)

        # Get just peaks not the sourse
        # Uses the window selector #
        windowName = 'Valid region selector'
        win = cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 800, 600)
        r = cv2.selectROI(windowName=windowName, img=img_array)
        img_array = make_pixels_black(img_array, r,255)
        img2_temp=img2
        img2_temp=img2_temp*0
        img2 = swap_pixel_data(img2_temp, img2,r)
        cv2.destroyWindow(windowName)
        new_img = img-img2

        # Get max value location
        max_val = np.amax(new_img)
        rows,cols = new_img.shape
        peaks = get_max_locs(new_img, rows, cols, max_val)

        # From top peak find the rest
        new_img = manual_box_finder(new_img , peaks[0], peaks[-1], 5)

        #cv2.imwrite( "manual_box.tif", new_img)
        cv2.imshow('image', new_img)
        cv2.waitKeyEx()


    #todo: look at this
    def peak_finder(p, u, v, gamma):
        i,j = np.argmax(p)
        max_p = np.max(p)
        thresh = 1
        while max_p < thresh:
            u_0 = u(i,j)
            v_0 = v(i,j)
            q =  np.sqrt((u-u_0)**2 +(v-v_0)**2)
            rho = q < gamma
            #p = np.where(q<gamma)[p < gamma] = 0
            p[rho] = 0
            i,j = np.argmax(p)
            max_p = np.max(p)

        return (u(i,j), v(i,j))

    def intensity(self, peaks, fiber_pos, b):
        x, y = fiber_pos
        x0 = 107
        y0 = 96
        gamma = 10
        window = 500
        u, v = np.meshgrid(np.arange(window), np.arange(window))
        r = math.sqrt((x-x0)**2 +(y-y0)**2)
        A = math.exp(-(r/(2*gamma))**2)
        z = b * A
        q = []
        q_gaus = []
        p = 0
        for i in range(len(peaks)):
            u0, v0 = peaks[i]
            q.append( np.sqrt((u-u0)**2 +(v-v0)**2))
            q_gaus.append(z * np.exp(-(q[i]/(2*gamma))**2))
            p = p + q_gaus[i]
        plt.pcolor(p)
        plt.show()
        #print(np.max(p))
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
                I0.append((I[n], (x,y)))
        return max(I0)[1]

    def prob_2(self, attin, peaks):
        I = []
        fiber_pos = (107, 96)
        for i in range(len(attin)):
            #img = cv2.imread(imgs[i])
            b = attin[i]
            # peaks stored as tuple
            I.append(self.intensity(peaks, fiber_pos, b))
        # plot
        '''for j in range(len(attin)):
            x = I[j]
            plt.plot(x, attin)'''
        '''To do: find slopes'''

        return I


    ## Simulates lights if fiber is near in-grating ##
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
        peaks = [(9,216), (22,216),(35,214), (48,214), (63,211), (75,211), (89,208), (103, 208), (116,205), (129,205), (143, 202), (156, 202), (169, 199), (183, 198), (197, 194), (210, 194)]
        for i in range(len(peaks)):
            y, x = peaks[i]
            light_img = cv2.circle(light_img,(x,y),2,(0,255,255),2)
        return light_img
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

if __name__ == '__main__':
    sim = SimulatedEnvironment()

    '''To run the peak finder'''
    #peaks = sim.set_peaks('SUWG01_01-lamp.tif', 'SUWG01_01-0dB.tif')

    '''To run intensity problems'''
    #imgs = ['SUWG01_01-20dB.tif','SUWG01_01-13dB.tif','SUWG01_01-10dB.tif', 'SUWG01_01-0dB.tif']
    attin = [0.01, 0.05, 0.1, 1]
    peaks = [(22,216), (129,205), (210,194)]

    #sim.prob_1(0, peaks)
    sim.prob_2(attin, peaks)

    #SUWG01_01-lamp.tif
    '''To run the fiber simulator'''
    #if len(sys.argv) > 1:
    #    sim.key_control(str(sys.argv[1]))


