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


def huffman_enconding(text):
    from collections import Counter
    import heapq

    class Node:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            return self.freq < other.freq

    def build_huffman_tree(text):
        frequency = Counter(text)
        heap = [Node(char, freq) for char, freq in frequency.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            merged = Node(None, left.freq + right.freq)
            merged.left = left
            merged.right = right

            heapq.heappush(heap, merged)

        return heap[0]

    def build_huffman_code(node, prefix="", code={}):
        if node:
            if node.char:
                code[node.char] = prefix
            build_huffman_code(node.left, prefix + "0", code)
            build_huffman_code(node.right, prefix + "1", code)

        return code

    root = build_huffman_tree(text)
    code = build_huffman_code(root)

    return "".join([code[char] for char in text])


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