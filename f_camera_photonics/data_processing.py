import os
from glob import iglob
import numpy as np
import matplotlib.pyplot as plt

from f_camera_photonics.peak_finder import main, load_output, save_output


### Batch processing on directories ###

def process_directory(dirname='', box_spec=None, new_only=False, glob='*[(.tif)(.tiff)]', **config_overrides):
    pathpattern = os.path.join(dirname, glob)
    all_json_files = []
    for fn in iglob('*.json'):
        filebase = os.path.splitext(os.path.basename(fn))[0]
        all_json_files.append(filebase)
    for fn in iglob(pathpattern):
        filebase = os.path.splitext(os.path.basename(fn))[0]
        if new_only:
            if filebase in all_json_files:
                print('Not redoing', filebase)
                continue
        pout = main(fn, box_spec, **config_overrides)
    consolidate_data(dirname)


def consolidate_data(dirname=''):
    all_pout = dict()
    pathpattern = os.path.join(dirname, '*.json')
    if os.path.exists(os.path.join(dirname, 'all_data.json')):
        os.remove(os.path.join(dirname, 'all_data.json'))
    for fn in iglob(pathpattern):
        filebase = os.path.splitext(os.path.basename(fn))[0]
        pout = load_output(fn)
        all_pout[filebase] = pout
    save_output(all_pout, os.path.join(dirname, 'all_data.json'))


### File naming convention ###

def index_to_name(index_dict):
    ''' Converts dict(x=3, y=5) to "x3y5".
        Keys will be sorted alphanumerically
    '''
    name = ''
    for dim_name in sorted(index_dict.keys()):
        name += dim_name
        ind = index_dict[dim_name]
        if type(ind) is not int or ind > 9 or ind < 1:
            raise ValueError('Bad index: {}. Only integers 1-9 are supported.'.format(ind))
        name += str(ind)
    return name


def name_to_index(name):
    ''' Converts "x3y5.tif" or "x3y5" to dict(x=3, y=5)

        raises ValueError if the naming convention does not work.

        Limited processor as of now: supports single digit only
    '''
    index_dict = dict()
    index_variables = list('wxyz')
    name = name.split('.')[0]  # remove suffix

    # Check naming convention
    conforming = True
    for char in name:
        if not char.isdigit() and not char in index_variables:
            conforming = False
    if not name[-1].isdigit():
        conforming = False
    if not conforming:
        raise ValueError('Name "{}" does not conform to the naming convention: "w1x3y2".'.format(name))

    # Parse the values
    for ivar in index_variables:
        try:
            pos = name.index(ivar)
        except ValueError:
            continue
        ival = name[pos + 1]
        index_dict[ivar] = int(ival)
    return index_dict


### Data analysis over multiple files ###

def calc_shape(data_dict):
    ''' Turn the dictionary of data entries into grid-like information,
        such as ((4, 5), ['x', 'y']) for a 4x5 grid calling its parameters "x" and "y".

        data_dict is what you would get from ``load_output('all_data.json')``
    '''
    dim_sizes = None
    for data_name, data_entry in data_dict.items():
        index_dict = name_to_index(data_name)

        if dim_sizes is None:
            dim_sizes = dict((k, 0) for k in index_dict.keys())
        elif set(index_dict.keys()) != set(dim_sizes.keys()):
            raise ValueError('Inconsistent number or name of parameter dimensions in {}'.format(data_name))

        for dim_name in dim_sizes.keys():
            dim_sizes[dim_name] = max(dim_sizes[dim_name], index_dict[dim_name])
    dim_names = sorted(dim_sizes.keys())
    shape = tuple(dim_sizes[dim] for dim in dim_names)
    return shape, dim_names


def default_entry_to_scalar(data_entry):
    return data_entry['Normalized Power'][1]


def convert_to_array(data_dict, entry_to_scalar=default_entry_to_scalar):
    ''' Populate an array with scalars based on a dictionary of data.
        data_dict is what you would get from ``load_output('all_data.json')``

        The shape of the array is determined by the grid signified by the data names.

        The scalars are derived using the ``entry_to_scalar`` argument,
        which is a function that changes often based on the type of device.
        Missing elements are NaN.
    '''
    shape, dim_names = calc_shape(data_dict)
    array_vals = np.empty(shape)
    array_vals[:] = np.nan
    for data_name, data_entry in data_dict.items():
        index_dict = name_to_index(data_name)
        index = tuple(index_dict[dim] - 1 for dim in dim_names)
        value = entry_to_scalar(data_entry)
        array_vals[index] = value
    return array_vals


def plot_1d(data_array):
    plt.plot(data_array, '-.')


def plot_2d(data_array):
    ''' Wrapper for ``pcolor`` with options that are good for gridded integer domains
    '''
    fi, ax = plt.subplots()
    colorvals = ax.pcolor(data_array.T)
    # Put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data_array.shape[0]) + 0.5)
    ax.set_yticks(np.arange(data_array.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(data_array.shape[0]) + 1)
    ax.set_yticklabels(np.arange(data_array.shape[1]) + 1)
    # Label and color
    cbar = plt.colorbar(colorvals)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.show()
