# Gets a picture from the default remote computer and displays it
from f_camera_photonics import cvshow
from f_camera_photonics.tcp_link import remote_call, unpack_image

img_serial = remote_call('capture', avgcnt=1)
cvshow(unpack_image(img_serial))