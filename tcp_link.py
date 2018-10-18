import zmq
from socket import getfqdn
import json
from types import FunctionType
import cv2
from functools import wraps

from component_capture import single_shot, video_mean

# Commands are tokened by the name of the function
_available_commands = dict()
_raw_returners = set()
def tcp_command(json_codec=True):
    @wraps(func)
    def tcp_register(func):
        global _available_commands
        global _raw_returners
        _available_commands[func.__name__] = func
        if not json_codec:
            _raw_returners.add(func.__name__)
        return func
    return tcp_register


@tcp_command()
def ping():
    return 'Hello there'

@tcp_command(json_codec=False)
def capture(nframes=1):
    img = single_shot()
    img_serial = cv2.imencode('.png', img)[1].tobytes()
    return img_serial

@tcp_command()
def kill():
    ''' Windows doesn't like keyboard interrupts '''
    raise RuntimeError('Remote server kill')


## command and control layer.
# Converts between arg/kwarg-like objects and TCP messages.
# Calls the server-side functions

def pack_command(cmd_name, *args, **kwargs):
    if type(cmd_name) is FunctionType:
        cmd_name = cmd_name.__name__
    if cmd_name not in _available_commands.keys():
        raise KeyError('No command named {}'.format(cmd_name))
    command_struct = (cmd_name, args, kwargs)
    return json.dumps(command_struct).encode()

def parse_command(msg_bytes):
    cmd_name, args, kwargs = json.loads(msg_bytes.decode())
    func = _available_commands[cmd_name]
    resp = func(*args, **kwargs)
    if cmd_name in _raw_returners:
        return resp
    else:
        return json.dumps(resp).encode()

def unpack_response(resp_bytes):
    return json.loads(resp_bytes.decode())


## Process layer

def run_server(port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    print('Starting camera photonics server')
    print('FQDN =', getfqdn())
    print('PORT =', port)
    socket.bind("tcp://*:{}".format(port))  # * means localhost

    while True:
        message = socket.recv()
        print("Received request: %s" % message)
        response = parse_command(message)
        socket.send(response)


def remote_call(cmd_name, *args, address='686NAM3560B.campus.nist.gov', port=5555, **kwargs):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://{}:{}'.format(address, port))

    if type(cmd_name) is FunctionType:
        cmd_name = cmd_name.__name__

    socket.send(pack_command(cmd_name, *args, **kwargs))
    if cmd_name == 'kill':
        return None
    elif cmd_name in _raw_returners:
        return socket.recv()
    else:
        return unpack_response(socket.recv())

if __name__ == '__main__':
	run_server()

