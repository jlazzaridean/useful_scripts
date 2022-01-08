from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tf

def calc_mean_bin(decay, axis=0):
    # decay is an N dimensional array where slowest dimension is TCSPC time by default
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
im_path = current_dir / '2021-10-27_img1_frame1_test.tif'
im = tf.imread(im_path) # 3D numpy array. dimensions are T, Y, X

# some timing metadata
laser_period_ns = 25
ADC_resolution = 400

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
                                   laser_period_ns / ADC_resolution)

# save the output to disk
tf.imwrite('mean_arrival_test.tif', output_image.astype('float32'), imagej=True,
           metadata={'axes': 'CYX'})
