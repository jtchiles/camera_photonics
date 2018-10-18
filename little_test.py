from f_camera_photonics import cvshow
from tcp_link import remote_call, unpack_image

img_serial = remote_call('capture', avgcnt=1)
cvshow(unpack_image(img_serial))