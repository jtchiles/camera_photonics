# Camera Photonics
High-throughput metrology for integrated photonics. Invented at NIST Boulder by Jeff Chiles.

Other software contributions by Jacob Melonis and Alex Tait.

## Image processing
The main point of this project. TODO

## Data processing
TODO

## Capturing images from code
### Experimental configuration
Pretty much any camera has an analog component output. This is being converted to USB by an SVID2USB2NS from StarTech. USB is going into a Windows PC because there are no Linux drivers. (There are Mac OS ones though.)

The driver is [here](https://sgcdn.startech.com/005329/media/sets/SVID2USB2_Drivers/[eMPIA%20EM28xx]%20Windows%20USB%20Video%20Capture%20Cable.zip). Installation process is weird. Instructions [here](https://www.startech.com/faq/SVID2USB2_Install_Advanced).

If successful, you will be able to grab images within NI-MAX.

### API interface
The gathering happens in the `component_capture.py` file using the `cv2` package. It's actually pretty straightforward. Usually, the StarTech thing is on `camera_port` 0, but that might change in principle. To get a list of frames as np.ndarray's, call `get_frames`. You can automatically average them - this is important when operating remotely to avoid the long delay in transmitting the file over network.

### Remote operation
This happens in the `tcp_link.py` file using the `Py0MQ`package. You must have this file present in both the serving machine and the client maching. I recommend having the git project cloned on each so that their versions stay synchronized.

Set the address and port either with
```python
import f_camera_photonics.tcp_link
f_camera_photonics.tcp_link.remote_address_default = 'XXX.campus.edu'
```
or as an argument to `remote_call`.

The protocol is pretty much an arbitrary command/query structure; however, running arbitrary code on the server is not allowed. Valid commands must have the `@tcp_command` decorator.

To launch the server, run
```bash
python tcp_link.py
```
God help you if you want to choose which python version you are using on Windows. For that reason, this command works with python 2 or 3.
