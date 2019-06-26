''' How to check that this is working. Note some values are hard-coded now (fiber ports).

    Environment: Run sim_demo(). You should see a window with a picture of a chip and 8 ports.
    Pressing WASD and 1-9 will change the image. When you click, the coordinates will print.

    Runner: Run sim_demo(True).
'''


import numpy as np
import cv2
import sys
import time
import math
import matplotlib.pyplot as plt
from f_camera_photonics.peak_finder import cvshow, pick_ports, PortArray
from f_camera_photonics.attenuator_driver import atten_lin, atten_db
from f_camera_photonics.component_capture import single_shot
from f_camera_photonics.tcp_link import remote_call, unpack_image

default_size = (200, 200)
def get_domain(size=None):
    # returns arrays representing x, y at every grid point, depending on size
    if size is None:
        size = default_size
    u = np.arange(size[0])
    v = np.arange(size[1])
    U, V = np.meshgrid(v, u)
    return U, V


peak_fwhm = 5
def gauss(distance, fwhm=peak_fwhm):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    return np.exp(- (distance ** 2 / 2 / sigma ** 2))


def gauss_mesh(center, fwhm=peak_fwhm, size=None):
    U, V = get_domain(size=size)
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

    def set_atten_db(self, atten):
        print('mark 1')
        self.atten = 10 ** (-atten / 10)

    def snap(self):
        # returns an image just like the one you get from the camera
        frame = self.device.background.copy()
        # input fiber spot
        frame += self.atten * gauss_mesh(center=self.fiber_pos, size=frame.shape)

        # how far off is the fiber from in_gc? – get a factor
        dfiber = self.fiber_pos - self.device.in_gc_pos
        alignment_factor = gauss(np.sqrt(np.sum(dfiber ** 2)))

        # what are port factors - list of factors
        for port_pos, port_intensity in self.device.out_gc_list:
            port_spot = gauss_mesh(center=port_pos, size=frame.shape)
            port_spot *= self.atten * alignment_factor * port_intensity
            frame += port_spot
        # clip it
        frame = np.clip(frame, 0, 1)
        return frame

    def show(self, mouse_callback=None):
        # shows the current state
        cvimg = self.snap()
        return cvshow(cvimg)

    def interactive(self):
        # listen for WASD and number keys
        # NOTE: this is not meant to be used in a real simulation activity, just for testing
        while True:
            cvimg = self.snap()
            # keyval = self.show(cvimg, circle_follow)
            keyval = better_show(cvimg, mouse_callback=lambda w, i: array_follow(w, i, np.array([24,20]), 8))

            if isinstance(keyval, str):
                if keyval == 'W':
                    self.move_fiber_by(0, -1)
                elif keyval == 'S':
                    self.move_fiber_by(0, 1)
                elif keyval == 'A':
                    self.move_fiber_by(-1, 0)
                elif keyval == 'D':
                    self.move_fiber_by(1, 0)
                elif keyval == 'Space':
                    break
            elif isinstance(keyval, int):
                self.set_atten_lin(float(keyval) / 9.0)

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
            cv2.circle(img_copy, center, 7, (255, 64, 64), 1)
            cv2.imshow(windowName, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            print('Point is picked', x, y)
            print('Press any key')
            selection_data = (x, y)
            # cv2.destroyWindow(windowName)
        else:
            print('Other event: ', event)
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


def better_show(cvimg, windowName='better_img', mouse_callback=None):
    ''' Several extra features
        1. Reasonable window size
        2. Mouse callbacks that are functions to do stuff when mouse is moved or pressed
        3. Key press handling

        It does not destroy the window
    '''
    global selection_data
    # big = cv2.resize(cvimg, (0,0), fx=3, fy=3)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 700, 700)
    if mouse_callback is not None:
        # print('Callback has been set')
        cv2.setMouseCallback(windowName, mouse_callback(windowName, cvimg))
    selection_data = None
    cv2.imshow(windowName, cvimg)
    keycode = cv2.waitKey(0)
    # keyval = chr(keycode & 0xFF)
    try:
        keyval = keyboard[keycode]
    except KeyError:
        keyval = None
    # cv2.destroyWindow(windowName)
    return keyval


keyboard = {119: 'W', 97: 'A', 115: 'S', 100: 'D'} # movement
keyboard[32] = 'Space' # exit
for i in range(10):  # attenuation
    keyboard[i + 48] = i


class Runner(object):
    def __init__(self, simulator):
        if not isinstance(simulator, SimEnviron2) and simulator not in ['RealLife', 'RemoteLife']:
            raise TypeError('Improper environment: {}'.format(simulator))
        self.sim = simulator
        self.background = None
        self.port_array = None
        # self.selection_data = dict()

    def move_fiber_by(self, dx, dy):
        if self.sim == 'RealLife':
            print('No translation stage')
            return
        elif self.sim == 'RemoteLife':
            print('No translation stage')
            return
        # elif isinstance(self.sim, SimEnviron2):
        else:
            return self.sim.move_fiber_by(dx, dy)

    def move_fiber_to(self, x, y):
        if self.sim == 'RealLife':
            print('No translation stage')
            return
        elif self.sim == 'RemoteLife':
            print('No translation stage')
            return
        # elif isinstance(self.sim, SimEnviron2):
        else:
            return self.sim.move_fiber_to(x, y)

    def set_atten_lin(self, atten):
        atten = min(atten, 1e-12)
        attendb = -10 * np.log10(atten)
        self.set_atten_db(attendb)

    def set_atten_db(self, atten=None):
        if self.sim == 'RealLife':
            return atten_db(atten)
        elif self.sim == 'RemoteLife':
            return remote_call('attenuate', atten=atten)
        # elif isinstance(self.sim, SimEnviron2):
        else:
            return self.sim.set_atten_db(atten)

    def raw_snap(self):
        if self.sim == 'RealLife':
            return single_shot()
        elif self.sim == 'RemoteLife':
            return unpack_image(remote_call('capture'))
        # elif isinstance(self.sim, SimEnviron2):
        else:
            return self.sim.snap()

    def snap(self, remove_background=False):
        raw = self.raw_snap()
        if remove_background:
            if self.background is None:
                raise ValueError('background has not been taken')
            return raw - self.background
        else:
            return raw

    def show(self, remove_background=True):
        img_diff = self.snap(remove_background=remove_background)
        keyval = better_show(img_diff, 'Space to close')
        time.sleep(1)
        cv2.destroyWindow('Space to close')

    def acquire_background(self):
        # remove the background
        self.set_atten_lin(0)
        self.background = self.raw_snap()
        # self.set_atten_lin(1)
        # img_on = self.snap()
        # img_diff = img_on - img_off

    def interactive_pick_port(self):
        # take a new image
        img_diff = self.snap(remove_background=True)
        keyval = better_show(img_diff, 'Click the first port', mouse_callback=circle_follow)
        port1 = selection_data
        all_ports = PortArray()
        all_ports.add_port(port1[0], port1[1])
        return all_ports

    def interactive_pick_array(self, nports=16):
        # Present the user with the peak picking step for multiple ports
        img_diff = self.snap(remove_background=True)
        better_show(img_diff, 'Click the first port', mouse_callback=circle_follow)
        port1 = selection_data
        better_show(img_diff, 'Click the last port', mouse_callback=lambda w, i: array_follow(w, i, port1, nports))
        port8 = selection_data
        dxdy = np.subtract(port8, port1) / (nports - 1)
        all_ports = PortArray()
        for iPort in range(nports):
            this_refport = np.array(port1) + np.array(dxdy) * iPort
            all_ports.add_port(this_refport[0], this_refport[1])
        return all_ports

    def interactive_pick_interleaved_array(self, nexperiments=8):
        all_ports = self.interactive_pick_array(nports=nexperiments * 2)
        ref_ports = PortArray()
        test_ports = PortArray()
        for iPortPair in range(nexperiments):
            pinfo = all_ports[iPortPair]
            ref_ports.add_port(pinfo[0], pinfo[1])
            pinfo = all_ports[iPortPair + 1]
            test_ports.add_port(pinfo[0], pinfo[1])
        return ref_ports, test_ports

    def set_live_ports(self, port_array):
        self.port_array = port_array

    def interactive(self):
        # Measure vs attenuation
        # attendb_arr = [-30, -20, -10, -3, 0]
        # import pdb; pdb.set_trace()
        atten_arr = np.linspace(1e-3, 1, 9)
        ref_powers = np.zeros((nports, len(atten_arr)))
        test_powers = np.zeros((nports, len(atten_arr)))
        for iAtten, atten in enumerate(atten_arr):
            self.set_atten_lin(atten)
            image = self.snap() - img_off
            ref_powers[:, iAtten] = ref_ports.calc_powers(image)
            test_powers[:, iAtten] = test_ports.calc_powers(image)
        plt.plot(atten_arr, ref_powers[-1, :])
        plt.show()


class AlignmentRunner(Runner):
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


def real_demo():
    runner = Runner('RemoteLife')
    runner.interactive()

def sim_demo(use_runner=False):
    bg = cv2.imread('example_image.tif', -1).astype('float')
    bg /= 2 ** 12
    dev = DynDevice(background=bg)
    sim = SimEnviron2(dev)

    if use_runner:
        runner = Runner(sim)
        runner.interactive()
    else:
        sim.interactive()

def get_runner(level=0):
    # See if it runs without error
    bg = cv2.imread('example_image.tif', -1).astype('float')
    bg /= 2 ** 12
    dev = DynDevice(background=bg)
    sim = SimEnviron2(dev)
    runner = Runner(sim)

    if level >= 1:
        # Background removal
        runner.acquire_background()
        runner.set_atten_db(0)
        if level == 1:
            runner.show()
    if level >= 2:
        # Interactive peaks
        new_ports = runner.interactive_pick_array()
        new_ports.sort_by('y')
        runner.port_array = new_ports
    if level >= 3:
        # Power matrix
        attens = [20, 10, 0]
        powers = np.zeros((len(attens), len(new_ports)))
        for iAtten, atten in enumerate(attens):
            runner.set_atten_db(atten)
            image = runner.snap(remove_background=True)
            runner.port_array.calc_powers(image)
            powers[iAtten, :] = runner.port_array.P_vec
        print(powers)
        plt.plot(attens, powers)
        return powers
    return runner


if __name__ == '__main__':
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

    sim_demo()


