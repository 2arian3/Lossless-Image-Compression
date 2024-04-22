import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    get_image_as_array,
    get_hilbert_curve_points,
    huffman_decoding,
    huffman_encoding
)


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_dir>")
        sys.exit(1)
    
    image_dir = sys.argv[1]

    image = get_image_as_array(image_dir)
    image = image[::-1, :, :] # Flip the image vertically
    image_w, image_h, image_c = image.shape

    # order = int(np.log2(max(image_w, image_h)))
    order = 8
    N = 2**order

    h_x, h_y = get_hilbert_curve_points(order=order, norm=False, offset=0)
    h_x = h_x.astype(int)
    h_y = h_y.astype(int)

    linearized_image = np.zeros((N * N, image_c), dtype=np.uint8)
    for i in range(N * N):
        linearized_image[i, :] = image[h_y[i], h_x[i], :]





if __name__ == "__main__":
    main()