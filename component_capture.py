# Trying to grab images out of cv2 with the USB cam
import cv2
import os
from contextlib import contextmanager

camera_port = 0

@contextmanager
def open_camera():
    camera = cv2.VideoCapture(camera_port)
    yield camera
    del(camera)


## Low level conditioning
# Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30

def get_frames(nframes=1):
    with open_camera() as camera:
        for i in range(ramp_frames):
            camera.read()
        frame_list = []
        for i in range(nframes):
            _, img = camera.read()
            frame_list.append(img)
    return frame_list


def single_shot():
    return get_frames(1)[0]


def average(nframes=2):
    stack = np.array(get_frames(nframes))
    return np.mean(stack, axis=2)


if __name__ == '__main__':
    from f_camera_photonics import cvshow
    cvshow(single_shot())
