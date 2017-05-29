import argparse
from PIL import Image
import numpy as np


def process(out, cb, cr):
    out_img_y = out
    # out_img_y *= 255.0
    # out_img_y = out_img_y.clip(0, 255)
    # out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

parser = argparse.ArgumentParser()
parser.add_argument('-orig_img')
parser.add_argument('-make_img')
# parser.add_argument('-scale')
args = parser.parse_args()

img = Image.open(args.orig_img).convert('YCbCr')
y, cb, cr = img.split()

hr_img = Image.open(args.make_img).convert('L')
# hr_img = Image.open(args.orig_img).convert('L')
# y_hr, cb_hr, cr_hr = hr_img.split()

# np_hr_img = np.array(y_hr)
np_hr_img = np.array(hr_img)
hr_color = process(hr_img, cb, cr)
hr_color.save('/Users/boiarov/data/SR/color.png')



