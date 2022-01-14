from pathlib import Path
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# This function generates a lifetime-intensity overlay image from a mean
# arrival image (*.tif), where the first channel is the photon counts
# and the second channel is the lifetime in ns.

# TODO: enable conversion to pH instead of lifetime
# TODO: perhaps warn when scale is clipping?

current_dir = Path.cwd()
output_path = current_dir 
im_path_list = [current_dir / 'mean_arrival_test.tif']

scale_bar = False
size_x_um = 128 * 1.06
scale_bar_um = 20
scale_bar_fraction = scale_bar_um / size_x_um

for i, im_path in enumerate(im_path_list):
    print('Image %d' % i)
    im = tf.imread(im_path) # C, Y, X

    # define the ranges for the two maps
    max_count = np.max(im[0, :, :], axis=(0, 1))
    tau_range = [2.4, 2.9] # units in the input images are ns
    print('Scaling intensity from 0 to %d' % max_count)
    print('Scaling ns lifetime from %0.2f to %0.2f' % (tau_range[0], tau_range[1]))
    
    fig1 = plt.figure()
    ax1 = plt.axes([0, 0, 1, 1])

    # use photons to make transparency map
    alpha_img = im[0, :, :] / max_count
    alpha_img[np.isnan(im[1, :, :])] = 0

    # add on the intensity data
    ax1.imshow(im[0, :, :], cmap='Greys_r')
    # overlay the lifetime data with the intensity as transparency
    ax1.imshow(im[1, :, :], alpha=alpha_img, vmin=tau_range[0], vmax=tau_range[1],
               cmap='turbo_r')
    norm_scale = colors.Normalize(vmin=tau_range[0], vmax=tau_range[1])
    fig1.colorbar(cm.ScalarMappable(norm=norm_scale, cmap='turbo_r'), ax=ax1)

    plt.text(im.shape[-1] + im.shape[-1]*0.2, im.shape[-2] / 2,
             'Lifetime (ns)', rotation=90, color='k', fontsize=16, va='center')

    # add on a scale bar
    if scale_bar:
        assert scale_bar_fraction < 0.3
        plt.axhline(y=im.shape[-2]*0.95, xmin=0.7,
                    xmax=0.7 + scale_bar_fraction, linewidth=2,
                    color='#FFFFFF', zorder=1000)
        plt.text((0.7 + scale_bar_fraction/2) * im.shape[-1], im.shape[-2]*0.9,
                 r'%d $\mu$m'%scale_bar_um, color='#FFFFFF', fontsize=16,
                 ha='center', va='top')

    # assorted clean-up
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # save the frame to .pdf
    out_name = im_path.stem + '.pdf'
    plt.savefig(output_path / out_name, dpi=300, bbox_inches='tight',
                pad_inches=0)
    plt.close()
