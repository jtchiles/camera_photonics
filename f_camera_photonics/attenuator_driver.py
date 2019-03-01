# It is a JDS HA9 optical attenuator

import serial
import visa
import time

usbport = 'COM4' # Prologix usb device
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
    if flag:
        print(inst.query('*IDN?'))
    return res, inst


def set_atten_lin(atten):
    pass

def set_atten_db(atten):
    pass

def enable(ena=True):
    pass