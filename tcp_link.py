import zmq
from socket import getfqdn
import json

from component_capture import single_shot, video_mean

# Commands are tokened by the name of the function
_available_commands = dict()
def tcp_command(func):
    global _available_commands
    _available_commands[func.__name__] = func
    return func


@tcp_command
def ping():
    return 'Hello there'

@tcp_command
def capture(nframes=1):
    img = single_shot()
    img_serial = cv2.imencode('.png', img)[1].tobytes()
    return img_serial

@tcp_command
def kill():
    ''' Windows doesn't like keyboard interrupts '''
    raise RuntimeError('Remote server kill')


## command and control layer.
# Converts between arg/kwarg-like objects and TCP messages.
# Calls the server-side functions

def pack_command(cmd_name, *args, **kwargs):
    if type(cmd_name) is not str:
        cmd_name = cmd_name.__name__
    if cmd_name not in _available_commands.keys():
        raise KeyError('No command named {}'.format(cmd_name))
    command_struct = (cmd_name, args, kwargs)
    return json.dumps(command_struct).encode()

def parse_command(msg_bytes):
    cmd_name, args, kwargs = json.loads(msg_bytes.decode())
    func = _available_commands[cmd_name]
    resp = func(*args, **kwargs)
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

    socket.send(pack_command(cmd_name, *args, **kwargs))
    if cmd_name != 'kill' and cmd_name.__name__ != 'kill':
        return unpack_response(socket.recv())

if __name__ == '__main__':
	run_server()

