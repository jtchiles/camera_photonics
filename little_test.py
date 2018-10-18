from tcp_link import remote_call
from f_camera_photonics import cvshow

img_str = remote_call('capture')
nparr = np.fromstring(img_str, np.uint8)
img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
cvshow(img_np)