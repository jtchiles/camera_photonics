# It is a JDS HA9 optical attenuator

import serial
import visa
import time
import numpy as np

usbport = 'COM5' # Prologix usb device
gpibport = 5 # Lakeshore thermometer port
dflag = True # Print debug messages flag


def prologix_startup(uport=usbport, gport=gpibport, flag=dflag):
    # Commands starting with '++' are for Prologix dongle itself
    ser = serial.Serial(uport, timeout = 0.5)
    if(flag):
        ser.write('++ver\n'.encode())
        print(ser.read(256).decode())
    ser.write('++mode 1\n'.encode()) # Set Prologix to CONTROLLER MODE
    ser.write('++addr {}\n'.format(gport).encode()) # Set Agilent GPIB address
    time.sleep(0.05)
    ser.write('++auto 1\n'.encode()) # Enable Prologix Read-after-write feature
    time.sleep(0.05)
    ser.close()


def get_visa_inst(uport=usbport, flag=dflag):
    res = visa.ResourceManager('@py') # '@py' loads PyVISA-py instead of NI-VISA
    inst =  res.open_resource('ASRL'+uport+'::INSTR')
    inst.timeout = 5000
    if flag:
        print(inst.query('*IDN?'))
    return inst


def atten_lin(atten=None):
    if atten is not None:
        atten = -10 * np.log10(atten)
    db_returned = atten_db(atten)
    return 10 ** (-db_returned / 10)


def atten_db(atten=None):
    # Controls the attenuation. If no argument, it queries the instrument.
    if atten is not None and (atten < 0 or atten > 100):
        raise ValueError('Attenuation {:.2f} is out of range [0--100]dB'.format(atten))
    inst = get_visa_inst()
    try:
        if atten is None:
            return float(inst.query('INP:ATT?').strip())
        else:
            inst.write('INP:ATT {}'.format(atten))
            return atten
    finally:
        inst.close()

def enable(ena=None):
    # Controls the blocking state. If no argument, it queries the instrument.
    inst = get_visa_inst()
    try:
        if ena is None:
            return inst.query('OUTP:STAT?').strip() == '1'
        else:
            inst.write('OUTP:STAT {}'.format(1 if ena else 0))
            return ena
    finally:
        inst.close()
