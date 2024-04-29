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
    entropy_ratio,
    burrows_wheeler_transform,
    inverse_burrows_wheeler_transform
)


def compress_image(image):
    image = image[::-1, :, :] # Flip the image vertically
    image_w, image_h, image_c = image.shape

    order = np.ceil(np.log2(max(image_w, image_h))).astype(int)
    N = 2**order

    expanded_image = np.zeros((N, N, image_c), dtype=image.dtype)
    expanded_image[:image_w, :image_h, :] = image

    h_x, h_y = get_hilbert_curve_points(order=order)
    h_x = h_x.astype(int)
    h_y = h_y.astype(int)

    linearized_image = np.zeros((N * N, image_c), dtype=np.uint8)

    for i in range(N * N):
        linearized_image[i, :] = expanded_image[h_y[i], h_x[i], :] 

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

    huffman_codes_size = 0
    for codes in huffman_codes:
        for code in codes.values():
            huffman_codes_size += len(code) + 8

    initial_img_size = image_w * image_h * image_c * 8
    compressed_img_size = sum([len(huffman_encoded_image[i]) for i in range(image_c)]) + huffman_codes_size

    print(f"Initial file size: {initial_img_size} bits")
    print(f"Compressed file size: {compressed_img_size} bits")
    print(f"Compression ratio: {initial_img_size / compressed_img_size:.4f}")
    print(f"Entropy ratio: {entropy_ratio(encoded_image):.4f}")

    return {
        "huffman_encoded_image": huffman_encoded_image,
        "huffman_codes": huffman_codes,
        "image_shape": (image_w, image_h, image_c),
    }


def decompress_image(image):
    huffman_encoded_image = image["huffman_encoded_image"]
    huffman_codes = image["huffman_codes"]
    image_w, image_h, image_c = image["image_shape"]

    decoded_image = [[] for _ in range(image_c)]
    for i in range(image_c):
        decoded_image[i] = huffman_decoding(huffman_encoded_image[i], huffman_codes[i])
        decoded_image[i] = move_to_front_decoding(decoded_image[i])

    decoded_image = np.array(decoded_image)

    order = np.log2(np.sqrt(decoded_image[0].shape[0])).astype(int)
    N = 2**order

    reconstructed_image = np.zeros((N, N, image_c), dtype=np.uint8)

    h_x, h_y = get_hilbert_curve_points(order=order)
    h_x = h_x.astype(int)
    h_y = h_y.astype(int)

    for i in range(N * N):
        reconstructed_image[h_y[i], h_x[i], :] = decoded_image[:, i]

    reconstructed_image = reconstructed_image[:image_w, :image_h, :]
    reconstructed_image = reconstructed_image[::-1, :, :]

    return reconstructed_image


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ["compress", "decompress"] or not (os.path.isdir(sys.argv[2]) or os.path.isfile(sys.argv[2])):
        print("Usage: python main.py compress/decompress <image_dir>")
        sys.exit(1)
    
    images_dir = sys.argv[2]
    images = [sys.argv[2]]


    if os.path.isdir(images_dir):
        images = [os.path.join(images_dir, image) for image in os.listdir(images_dir)]

    images.sort()


    if not os.path.exists("compressed_images"):
        os.makedirs("compressed_images")

    
    if not os.path.exists("decompressed_images"):
        os.makedirs("decompressed_images")


    if sys.argv[1] == "compress":
        for image in images:
            print(f"-------------------\nCompressing {image}")
            image_array = get_image_as_array(image)
            pickle_object = compress_image(image_array)
            print("-------------------")

            pickle.dump(pickle_object, open(f"compressed_images/compressed_{image.split('/')[-1]}.pickle", "wb"))


    elif sys.argv[1] == "decompress":
        for image in images:
            print(f"-------------------\nDecompressing {image}")
            compressed_image = pickle.load(open(image, "rb"))
            decompressed_image = decompress_image(compressed_image)
            result_dir = f"decompressed_images/decompressed_{image.split('/')[-1].rstrip('.pickle')}.png"
            print(f"Finished decompressing {image} and the result is saved as {result_dir}")
            print("-------------------")

            save_array_as_image(decompressed_image, result_dir)


if __name__ == "__main__":
    main()