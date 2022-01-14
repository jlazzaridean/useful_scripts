import struct
from pathlib import Path
import numpy as np
import tifffile as tf

# This script calculates the mean arrival time per pixel from a .BIN
# file (pre-histogrammed photon count file) generated from PicoQuant's
# SymPhoTime software.

def calc_mean_bin(decay, axis=0):
    # decay is an N dimensional array where slowest dimension is TCSPC
    # time by default
    s = np.ones(len(decay.shape), dtype='uint32')
    s[axis] = decay.shape[axis]
    bin_idx = np.arange(decay.shape[axis]).reshape(s)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (decay * bin_idx).sum(axis) / decay.sum(axis)
    return result

def bins_to_ns(mean_bin_image, zero_bin_idx, ns_per_bin):
    return (mean_bin_image - zero_bin_idx) * ns_per_bin

# read in the sample image
current_dir = Path.cwd()
in_path = current_dir / '2019-11-11_hek_gram_zeroConc'
out_path = current_dir / 'mean_arrival_images' / '2019-11-11_hek_gram_zeroConc'
im_path_list = [x for x in in_path.glob('*.bin') if 'glass' not in x.stem]
print('Found %d .bin files to process' % len(im_path_list))

for im_path in im_path_list:
    # let's get into this binary file and read it
    print('Opening file', im_path)
    with open(im_path, 'rb') as fid:
        nx = struct.unpack("i", fid.read(4))[0] # number of pixels in x
        ny = struct.unpack("i", fid.read(4))[0] # number of pixels in y
        pix_res_um = struct.unpack("f", fid.read(4))[0] # pixel res., um
        nt = struct.unpack("i", fid.read(4))[0] # number of TCSPC channels
        time_res_ns = struct.unpack("f", fid.read(4))[0] # timeper bin, ns
        print('%d by %d TCSPC image with %d time channels' % (nx, ny, nt))
        print('Spatial res. (um): %0.2e, Time/bin (ns): %0.2e' % (pix_res_um,
                                                                  time_res_ns))
        # im = np.zeros((nt, ny, nx), dtype='int32')
        im = np.array(struct.unpack((str(nt*ny*nx)+'i'),
                                    fid.read(nt*ny*nx*4))).reshape((ny, nx, nt))
        im = np.moveaxis(im, -1, 0) # now it's TYX

    # Calculate photons image and mean arrival time image. Make these
    # channels in a combined output_image
    output_image = np.zeros((2, im.shape[1], im.shape[2])) 
    np.sum(im, axis=0, out=output_image[0, :, :])
    mean_bin_image = calc_mean_bin(im)
    # now let's convert bins to ns. First, sum up the whole time series to
    # get a lower-noise determination of our "zero" point (laser pulse)
    time_trace = np.sum(im, axis=(1,2))
    # zero = point of greatest change, which is roughly the midpoint of the rise
    zero_idx = np.argmax(np.diff(time_trace)) - 1 
    output_image[1, :, :] = bins_to_ns(mean_bin_image, zero_idx,
                                       time_res_ns)

    # save the output to disk
    full_out = out_path / (im_path.stem + '.tif')
    tf.imwrite(full_out, output_image.astype('float32'),
               imagej=True, metadata={'axes': 'CYX'})
    print('Saving mean arrival image')
    print('\n')
