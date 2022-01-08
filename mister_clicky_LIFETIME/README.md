#mister_clicky_LIFETIME

This package is derived from marimar128's annotation_buddy. It uses a random forest classifier to segment objects in a supervised manner. This script in particular assumes that you are segmenting a hyperstack of two-channel TIFF images, where the first channel is the photon count and the second channel is the fluorescence lifetime in seconds (as it would be exported from SymPhoTime).

This version has been tested to work with python 3.9 and napari version 0.4.12.
