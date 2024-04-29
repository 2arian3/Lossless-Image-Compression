# Lossless-Image-Compression
This repo contains all the codes and documents for reproducing the results of EECS 6111 project.

## How to Use the Lossless Image Compression Tool

This tutorial will guide you through the process of using our Lossless Image Compression tool, including setup, compression, and decompression of images.

## Prerequisites

Before you begin, ensure you have Python and `virtualenv` installed on your machine.

## Getting Started

Follow these steps to set up the environment and run the tool.

### 1. Clone the Repository

Start by cloning the repository to your local machine:

    git clone https://github.com/2arian3/Lossless-Image-Compression
    cd Lossless-Image-Compression

### 2. Set Up the Virtual Environment

Create and activate a virtual environment:

    virtualenv venv
    source venv/bin/activate

### 3. Install Dependencies

Install the required Python packages:

    python3 -m pip install -r requirements.txt

## Running the Tool

### Compressing Images

To compress an image or a directory of images, use the following command:

    python3 main.py compress <image/s dir>

For example, to compress images in the `testsets/kodak` directory:

    python3 main.py compress testsets/kodak

### Decompressing Images

To decompress images, use the following command:

    python3 main.py decompress <file/s dir>

For example, to decompress images in the `compressed_images` directory:

    python3 main.py decompress compressed_images/

## Additional Notes

- Ensure that the paths provided to the commands are correct.
- The commands must be run from the root directory of the project unless paths are adjusted accordingly.

Feel free to contribute to improving the tool by submitting pull requests or filing issues on our repository.
