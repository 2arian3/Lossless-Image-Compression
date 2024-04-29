from PIL import Image
from collections import Counter
from heapq import heapify, heappop, heappush
import numpy as np


def get_image_as_array(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)

    if image.ndim == 2:
        # Add an extra dimension to make it 3D
        image = image[:, :, np.newaxis]

    return image


def save_array_as_image(array, image_path):
    image = Image.fromarray(array) if array.shape[2] == 3 else Image.fromarray(array[:, :, 0], mode="L")
    image.save(image_path)


def burrows_wheeler_transform(text):
    rotations = [text[i:] + text[:i] for i in range(len(text))]
    rotations.sort()

    return "".join([rotation[-1] for rotation in rotations])


def inverse_burrows_wheeler_transform(text):
    table = [""] * len(text)
    for i in range(len(text)):
        table = sorted([text[i] + table[i] for i in range(len(text))])
    return table[text.index("$")]


def move_to_front_encoding(img):
    alphabet = [i for i in range(256)]
    result = []

    for value in img:
        index = alphabet.index(value)
        result.append(index)
        alphabet.pop(index)
        alphabet.insert(0, value)

    return np.array(result)


def move_to_front_decoding(encoded_img):
    alphabet = [i for i in range(256)]
    result = []

    for index in encoded_img:
        char = alphabet[index]
        result.append(char)
        alphabet.pop(index)
        alphabet.insert(0, char)

    return np.array(result)


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


def run_length_decoding(encoded_text):
    result = []

    for count, char in encoded_text:
        result.append(char * count)

    return "".join(result)


def get_hilbert_curve_points(order, norm=False, offset=0.0):
    N = 2**order
    points = np.zeros((2, N * N))

    for i in range(N * N):
        U = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        V = np.array([offset, offset])
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


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_encoding(data):
  char_freq = Counter(data)

  nodes = [Node(char, freq) for char, freq in char_freq.items()]
  
  heapify(nodes)

  while len(nodes) > 1:
    left, right = heappop(nodes), heappop(nodes)

    merged = Node(None, left.freq + right.freq)
    merged.left = left
    merged.right = right
    heappush(nodes, merged)

  codes = {}

  def traverse(node, prefix="", code={}):
    if node is not None:
        if node.char is not None:
            code[node.char] = prefix
        traverse(node.left, prefix + '0', code)
        traverse(node.right, prefix + '1', code)
    return code

  codes = traverse(nodes[0], "")

  encoded_data = "".join(codes[char] for char in data)

  return encoded_data, codes


def huffman_decoding(data, codes):
    reverse_codes = {v: k for k, v in codes.items()}
    current_code = ""
    decoded_text = []

    for bit in data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_text.append(reverse_codes[current_code])
            current_code = ""
    
    return decoded_text


def compute_entropy(arr):
    values, counts = np.unique(arr, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def max_entropy(arr):
    unique_elements = len(np.unique(arr))
    if unique_elements > 1:
        return np.log2(unique_elements)
    else:
        return 1


def entropy_ratio(arr):
    entropy = compute_entropy(arr)
    max_entropy_value = max_entropy(arr)

    return entropy / max_entropy_value if max_entropy_value != 0 else 0