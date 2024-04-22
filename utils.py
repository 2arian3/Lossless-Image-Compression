from PIL import Image
import numpy as np


def get_image_as_array(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)

    return image


def burrows_wheeler_transform(text):
    rotations = [text[i:] + text[:i] for i in range(len(text))]
    rotations.sort()

    return "".join([rotation[-1] for rotation in rotations])


def move_to_front_transform(text):
    alphabet = [chr(i) for i in range(256)]
    result = []

    for char in text:
        index = alphabet.index(char)
        result.append(index)
        alphabet.pop(index)
        alphabet.insert(0, char)

    return result


def run_length_encoding(text):
    result = []
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1
        else:
            result.append((count, text[i - 1]))
            count = 1

    result.append((count, text[-1]))

    return result

def huffman_encoding(text):
    freq = {}
    for char in text:
        if char in freq:
            freq[char] += 1
        else:
            freq[char] = 1

    freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1])}

    nodes = list(freq.keys())
    while len(nodes) > 1:
        left = nodes.pop(0)
        right = nodes.pop(0)
        nodes.append((left, right))

    huffman_codes = {}
    def get_codes(node, code=""):
        if isinstance(node, str):
            huffman_codes[node] = code
        else:
            get_codes(node[0], code + "0")
            get_codes(node[1], code + "1")

    get_codes(nodes[0])

    encoded_text = "".join([huffman_codes[char] for char in text])

    return encoded_text, huffman_codes

def huffman_decoding(encoded_text, huffman_codes):
    huffman_codes = {v: k for k, v in huffman_codes.items()}
    decoded_text = ""
    code = ""
    for char in encoded_text:
        code += char
        if code in huffman_codes:
            decoded_text += huffman_codes[code]
            code = ""
    return decoded_text


def get_hilbert_curve_points(order, norm=True, offset=0.5):
    N = 2**order
    points = np.zeros((2, N * N))
    for i in range(N * N):
        U = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])   # four points under ORDER=1
        V = np.array([offset, offset])                   # offset: starting point
        for j in reversed(range(order)):
            index = i // 4**j % 4
            length = 2**j
            if index == 0:
                U[1], U[3] = U[3].copy(), U[1].copy() 
            elif index == 3:
                U[0], U[2] = U[2].copy(), U[0].copy()
            V += U[index] * length
        points[:, i] = V
    if norm:
        points /= N
    return points