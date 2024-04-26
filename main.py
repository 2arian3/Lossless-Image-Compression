import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    get_image_as_array,
    get_hilbert_curve_points,
    move_to_front_decoding,
    move_to_front_encoding,
    huffman_encoding,
    huffman_decoding,
    run_length_encoding,
    run_length_decoding,
    save_array_as_image,
)


def compress_image(image):
    image = image[::-1, :, :] # Flip the image vertically
    image_w, image_h, image_c = image.shape

    order = int(np.log2(max(image_w, image_h)))
    N = 2**order

    h_x, h_y = get_hilbert_curve_points(order=order)
    h_x = h_x.astype(int)
    h_y = h_y.astype(int)

    linearized_image = np.zeros((N * N, image_c), dtype=np.uint8)
    for i in range(N * N):
        linearized_image[i, :] = image[h_y[i], h_x[i], :]

    linearized_image_length = linearized_image.shape[0]

    # Move-to-front encoding for each channel
    encoded_image = np.zeros((linearized_image_length, image_c), dtype=np.uint8)
    for i in range(image_c):
        encoded_image[:, i] = move_to_front_encoding(linearized_image[:, i])

    # Huffman encoding for each channel
    huffman_encoded_image = [[] for _ in range(image_c)]
    huffman_codes = [[] for _ in range(image_c)]
    for i in range(image_c):
        huffman_encoded_image[i], huffman_codes[i] = huffman_encoding(encoded_image[:, i])

    initial_img_size = image_w * image_h * image_c * 8
    compressed_img_size = sum([len(huffman_encoded_image[i]) for i in range(image_c)]) + sum([len(huffman_codes[i]) for i in range(image_c)]) * 2 * 8

    print(f"Initial file size: {initial_img_size} bits")
    print(f"Compressed file size: {compressed_img_size} bits")
    print(f"Compression ratio: {initial_img_size / compressed_img_size:.2f}")

    # with open("compressed_image", "wb") as f:
    #     pickle.dump({
    #         "huffman_encoded_image": huffman_encoded_image,
    #         "huffman_codes": huffman_codes,
    #         "order": order
    #     }, f)


def decompress_image(image):
    huffman_encoded_image = image["huffman_encoded_image"]
    huffman_codes = image["huffman_codes"]
    order = image["order"]

    image_c = len(huffman_encoded_image)

    decoded_image = [[] for _ in range(image_c)]
    for i in range(image_c):
        decoded_image[i] = huffman_decoding(huffman_encoded_image[i], huffman_codes[i])
        decoded_image[i] = move_to_front_decoding(decoded_image[i])

    decoded_image = np.array(decoded_image)

    N = 2**order

    reconstructed_image = np.zeros((N, N, image_c), dtype=np.uint8)

    h_x, h_y = get_hilbert_curve_points(order=order)
    h_x = h_x.astype(int)
    h_y = h_y.astype(int)

    for i in range(N * N):
        reconstructed_image[h_y[i], h_x[i], :] = decoded_image[:, i]

    reconstructed_image = reconstructed_image[::-1, :, :]

    save_array_as_image(reconstructed_image, "decompressed_image.png")
    


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ["compress", "decompress"] or not os.path.exists(sys.argv[2]):
        print("Usage: python main.py compress/decompress <image_dir>")
        sys.exit(1)
    
    image_dir = sys.argv[2]

    if sys.argv[1] == "compress":
        image = get_image_as_array(image_dir)
        compressed_image = compress_image(image)


    elif sys.argv[1] == "decompress":
        compressed_image = pickle.load(open(image_dir, "rb"))
        decompressed_image = decompress_image(compressed_image)


if __name__ == "__main__":
    main()