import cv2
import PIL
import numpy as np
import PIL.ImageOps
from PIL import ImageEnhance, ImageFilter

def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def preprocess(image, contrast_value):
    image, alpha, beta = automatic_brightness_and_contrast(image, clip_hist_percent=5)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    pil_image = PIL.Image.fromarray(image)
    pil_image = pil_image.crop((0, 50, 350, 500))
    inverted_image = PIL.ImageOps.invert(pil_image)
    contrast = ImageEnhance.Contrast(inverted_image)
    im_output = contrast.enhance(contrast_value)
    im_output = im_output.filter(ImageFilter.SHARPEN)
    return im_output


def text_decompose(s):
    char_to_delete = '\x1d'
    s = s.replace(char_to_delete, "")

    gtin = s.partition("01")[2].partition("21")[0]
    sn = s.partition("21")[2].partition("91")[0]
    crypto = s.partition("91")[2]
    print('-' * 60)
    print("GTIN: ", gtin)
    print("SN: ", sn)
    print("CRYPTO: ", crypto)
    print('-' * 60)
