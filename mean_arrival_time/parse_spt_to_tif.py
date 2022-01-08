import os
import numpy as np
import picoquant_tttr_sin_corr as pq
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile as tf

current_dir = Path.cwd()
input_dir = current_dir
output_dir = current_dir 
image_path_list = [input_dir / '2021-10-27_01_5pointsDMSO_5pointsBaf2.ptu' ]
print('Found %d files to process' % len(image_path_list))

for j, file_path in enumerate(image_path_list):
    print("File %d, reading header..." % (j + 1), end='')
    tags = pq.parse_tttr_header(file_path, verbose=False)
    print("done")
    print("Num tags:", sum(len(tags[t]['values']) for t in tags))
    print("Num unique tags:", len(tags))
    print()
    # We loop over the frames one at a time:
    frames = pq.generate_picoharp_t3_frames(file_path, tags, verbose=False)
    # hardcode size for now. ns time, y, x. only save ch 2 of frame 1
    nt, ny, nx = [400, 256, 256]
    im = np.zeros((nt, ny, nx))
    for i, f in enumerate(frames):
        print("Parsing frame ", i, '... ', sep='', end='')
        parsed_frame = pq.parse_picoharp_t3_frame(
            records=f,
            tags=tags,
            verbose=True,
            show_plot=False,
            sin_corr_value=10, # equal to the scan speed / 10
            sinusoid_correction=True) 
        print("done.")
        im_temp = pq.parsed_frame_to_histogram(parsed_frame, x_pix_per_bin=4,
                              y_pix_per_bin=4, t_pix_per_bin=4)
        im[:, :, :] = im_temp[:400, 1, :, :]

        break # only do the first frame for this demonstration

    print(im.shape, im.dtype)

    # save the output as .tif (with T dimension still intact)    
    output_name = '2021-10-27_img1_frame1_test.tif'
    print("Writing output %s" % output_name)
    print('\n')
    output_path = output_dir / output_name
    tf.imwrite(output_path, im.astype('float32'),imagej=True,
               metadata={'axes': 'TYX'})

