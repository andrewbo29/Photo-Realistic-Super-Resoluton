import argparse
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-orig_img')
parser.add_argument('-make_img')
# parser.add_argument('-scale')
args = parser.parse_args()

img = Image.open(args.orig_img)
# img = img.rotate(270)

new_img = img.resize((int(img.size[0] / 2), int(img.size[1] / 2)), Image.BICUBIC)
new_img.save(args.make_img)