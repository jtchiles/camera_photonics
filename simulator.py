import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
from f_camera_photonics.peak_finder import cvshow, pick_ports, PortArray
from f_camera_photonics.attenuator_driver import atten_lin
from f_camera_photonics.component_capture import single_shot
from f_camera_photonics.tcp_link import remote_call, unpack_image

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
            # print(x, y)
            if self.is_pos_correct(x, y):
                new_img = self.move(x, y, img)
                new_img = self.create_light(new_img)
            #else new_img = self.move(x, y, img)
            cv2.imshow('image', new_img)
            key = cv2.waitKeyEx()
            # Up/W = 119
            print(key)
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



size = (200, 200)
def get_domain():
    # returns arrays representing x, y at every grid point, depending on size
    u = np.arange(size[0])
    v = np.arange(size[1])
    U, V = np.meshgrid(v, u)
    return U, V


peak_fwhm = 5
def gauss(distance, fwhm=peak_fwhm):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    return np.exp(- (distance ** 2 / 2 / sigma ** 2))


def gauss_mesh(center, fwhm=peak_fwhm):
    U, V = get_domain()
    du = U - center[0]
    dv = V - center[1]
    R = np.sqrt(du ** 2 + dv ** 2)
    return gauss(R, fwhm=fwhm)


def to_uint(arr):
    # 1.0 gets mapped to 2^16-1
    new_arr = np.copy(arr)
    new_arr *= 2 ** 16
    return new_arr.astype('uint16')


bases = np.array([215, 7])
diff = np.array([-1.5, 13.5])
gcs = []
for iPort in range(16):
    gcs.append((bases + iPort * diff, 3 if iPort % 2 == 0 else np.exp(-iPort/10)))

class DynDevice(object):
    in_gc_pos = None
    out_gc_list = None  # a list of tuples: (position, intensity)
    background = None

    def __init__(self, background=None):
        self.in_gc_pos = np.array([100, 108])
        self.out_gc_list = gcs
        global size
        if background is None:
            self.background = np.zeros(size)
        else:
            self.background = background
            size = np.shape(self.background)


class SimEnviron2(object):
    device = None
    atten = None
    fiber_pos = None

    def __init__(self, device):
        self.device = device
        self.atten = 1
        self.fiber_pos = np.array([100, 108])

    def move_fiber_by(self, dx, dy):
        self.fiber_pos += np.array([dx, dy])

    def move_fiber_to(self, x, y):
        self.fiber_pos = np.array([x, y])

    def set_atten_lin(self, att):
        self.atten = att

    def snap(self):
        # returns an image just like the one you get from the camera
        frame = self.device.background.copy()
        # input fiber spot
        frame += self.atten * gauss_mesh(center=self.fiber_pos)

        # how far off is the fiber from in_gc? – get a factor
        dfiber = self.fiber_pos - self.device.in_gc_pos
        alignment_factor = gauss(np.sqrt(np.sum(dfiber ** 2)))

        # what are port factors - list of factors
        for port_pos, port_intensity in self.device.out_gc_list:
            port_spot = gauss_mesh(center=port_pos)
            port_spot *= self.atten * alignment_factor * port_intensity
            frame += port_spot
        # clip it
        frame = np.clip(frame, 0, 1)
        return frame

    def show(self, mouse_callback=None):
        cvimg = self.snap()
        better_show(cvimg, mouse_callback)

    def interactive(self):
        # run the loop like key_control
        # listen for WASD and number keys
        while True:
            # new_img = cv2.circle(self.snap(), tuple(self.fiber_pos), 8, (255,0,0), 2)
            # self.show(circle_follow)
            self.show(lambda w, i: array_follow(w, i, np.array([644,20]), 8))
            keycode = cv2.waitKeyEx()
            try:
                keyval = keyboard[keycode]
            except KeyError:
                continue
            if isinstance(keyval, str):
                if keyval == 'W':
                    self.fiber_pos[1] -= 1
                elif keyval == 'S':
                    self.fiber_pos[1] += 1
                elif keyval == 'A':
                    self.fiber_pos[0] -= 1
                elif keyval == 'D':
                    self.fiber_pos[0] += 1
                elif keyval == 'Space':
                    break
            elif isinstance(keyval, int):
                self.atten = float(keyval) / 9.0

# These are mouse callbacks
selection_data = None
def circle_follow(windowName, img):
    def circle_follow_inner(event, x, y, flags, param):
        global selection_data
        if selection_data is not None:
            x, y = selection_data
        if event == cv2.EVENT_MOUSEMOVE:
            img_copy = img.copy()
            center = (x, y)
            cv2.circle(img_copy, center, 7, (255, 0, 0), 1)
            cv2.imshow(windowName, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            print('Point is picked', x, y)
            print('Press any key')
            selection_data = (x, y)
            # cv2.destroyWindow(windowName)
    return circle_follow_inner


def array_follow(windowName, img, anchor_coord, nports):
    def array_follow_inner(event, x, y, flags, param):
        global selection_data
        if selection_data is not None:
            x, y = selection_data
        if event == cv2.EVENT_MOUSEMOVE:
            img_copy = img.copy()
            center = (x, y)
            dxdy = np.subtract(center, anchor_coord) / (nports - 1)
            for iPort in range(nports):
                this_coord = tuple((anchor_coord + iPort * dxdy).astype(int))
                cv2.circle(img_copy, this_coord, 7, (255, 0, 0), 1)
            cv2.imshow(windowName, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            print('Point is picked', x, y)
            print('Press any key')
            selection_data = (x, y)
            # cv2.destroyWindow(windowName)
    return array_follow_inner


def better_show(cvimg, windowName='img', mouse_callback=None):
    global selection_data
    # big = cv2.resize(cvimg, (0,0), fx=3, fy=3)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 700, 700)
    if mouse_callback is not None:
        print('Callback has been set')
        cv2.setMouseCallback(windowName, mouse_callback(windowName, cvimg))
    selection_data = None
    cv2.imshow(windowName, cvimg)
    cv2.waitKey(0)
    cv2.destroyWindow(windowName)


keyboard = {119: 'W', 97: 'A', 115: 'S', 100: 'D'} # movement
keyboard[32] = 'Space' # exit
for i in range(10):  # attenuation
    keyboard[i + 48] = i


class Runner(object):
    def __init__(self, simulator):
        self.sim = simulator

    def set_atten_lin(self, atten=None):
        if self.sim == 'RealLife':
            return atten_lin(atten)
        elif self.sim == 'RemoteLife':
            return remote_call('attenuate', atten= -10 * np.log10(atten))
        elif isinstance(self.sim, SimEnviron2):
            return self.sim.set_atten_lin(atten)
        raise TypeError('Improper environment: {}'.format(self.sim))

    def snap(self):
        if self.sim == 'RealLife':
            return single_shot()
        elif self.sim == 'RemoteLife':
            return unpack_image(remote_call('capture'))
        elif isinstance(self.sim, SimEnviron2):
            return self.sim.snap()
        raise TypeError('Improper environment: {}'.format(self.sim))

    def hdr_peaks(): pass

    def autoalign(): pass

    @staticmethod
    def _pick_peaks(diff_img, threshold=.5):
        # descend from top until hitting threshold
        # hard coded: the number of bright reference ports (8)
        X = pick_ports(diff_img, nports=8, cfg=None)

    def adjust_range(self):
        self.set_atten_lin(1e-9)
        img_off = self.snap()
        atten_bounds = [0, 1]
        prev_max = 0
        for _ in range(100):
            this_atten = np.mean(atten_bounds)
            self.set_atten_lin(this_atten)
            img_diff = self.snap() - img_off
            this_max = np.max(img_diff)
            if abs(this_max - prev_max) < .01:
                break
            elif this_max > 0.99:
                atten_bounds[1] = this_atten
            elif this_max < 0.9:
                atten_bounds[0] = this_atten
            else:
                break
            prev_max = this_max
            print(this_atten, '-', np.max(img_diff))
            cvshow(img_diff)
        return this_atten

    def locate_peaks(self):
        # turns attenuation on and off, differences, finds peaks
        best_atten = self.adjust_range()
        self.set_atten_lin(best_atten)
        return _pick_peaks(img_diff)

    def interactive(self):
        # remove the background
        self.set_atten_lin(0)
        img_off = self.snap()
        self.set_atten_lin(1)
        img_on = self.snap()
        img_diff = img_on - img_off
        # Present the user with the peak picking step
        nports = 8
        if False:
            better_show(img_diff, 'Click the first port', mouse_callback=circle_follow)
            port1 = selection_data
            better_show(img_diff, 'Click the last port', mouse_callback=lambda w, i: array_follow(w, i, port1, nports))
            port8 = selection_data
        else:
            port1 = (216, 7)
            port8 = (195, 196)
        dxdy = np.subtract(port8, port1) / (nports - 1)
        # Now do the port structures
        ref_ports = PortArray()
        test_ports = PortArray()
        for iPortPair in range(nports):
            this_refport = np.array(port1) + np.array(dxdy) * iPortPair
            ref_ports.add_port(this_refport[0], this_refport[1])
            test_ports.add_port(this_refport[0] + dxdy[0]/2, this_refport[1] + dxdy[1]/2)
        # Measure vs attenuation
        # attendb_arr = [-30, -20, -10, -3, 0]
        atten_arr = np.linspace(1e-3, 1, 9)
        ref_powers = np.zeros((nports, len(atten_arr)))
        test_powers = np.zeros((nports, len(atten_arr)))
        for iAtten, atten in enumerate(atten_arr):
            self.set_atten_lin(10 ** (atten / 10))
            image = self.snap() - img_off
            ref_powers[:, iAtten] = ref_ports.calc_powers(image)
            test_powers[:, iAtten] = test_ports.calc_powers(image)
        import pdb; pdb.set_trace()
        plt.plot(atten_arr, ref_powers[-1, :])
        plt.show()


def real_demo():
    runner = Runner('RemoteLife')
    runner.interactive()


if __name__ == '__main__':
    # OLD
    # sim = SimulatedEnvironment()

    # '''To run the peak finder'''
    # #peaks = sim.set_peaks('SUWG01_01-lamp.tif', 'SUWG01_01-0dB.tif')

    # '''To run intensity problems'''
    # #imgs = ['SUWG01_01-20dB.tif','SUWG01_01-13dB.tif','SUWG01_01-10dB.tif', 'SUWG01_01-0dB.tif']
    # attin = [0.01, 0.05, 0.1, 1]
    # peaks = [(22,216), (129,205), (210,194)]

    # # sim.prob_1(0, peaks)
    # # sim.prob_2(attin, peaks)

    # #SUWG01_01-lamp.tif
    # '''To run the fiber simulator'''
    # if len(sys.argv) > 1:
    #    sim.key_control(str(sys.argv[1]))


    bg = cv2.imread('example_image.tif', -1).astype('float')
    bg /= 2 ** 12
    dev = DynDevice(background=bg)
    sim = SimEnviron2(dev)

    # NEW interactive
    # sim.interactive()

    runner = Runner(sim)
    # NEW diff plot
    runner.interactive()

