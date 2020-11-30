import numpy as np
from cv2 import cv2
import skimage.segmentation as segmentation
from skimage import img_as_ubyte

from skimage.filters import rank
from skimage.morphology import disk
from scipy import ndimage as ndi

from compare import compare_seg
import os


def load_image(path):
    image = cv2.imread(path, 0)
    return image

def load_image_rgb(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def increase_contrast(img, alpha=2):
    new_img = np.rint(alpha * (img.astype(np.int64) - 128) + 128)
    return new_img.clip(0, 255).astype(np.uint8)

def normalize(img):
    # grayscale
    if len(img.shape) == 2:
        return (255 * ((img - img.min()) / (img.max() - img.min()))).astype(np.uint8)
    # RGB
    else:
        norm_img = np.zeros(img.shape)
        for i in range(img.shape[2]):
            norm_img[:,:,i] = 255 * ((img[:,:,i] - img[:,:,i].min()) / (img[:,:,i].max() - img[:,:,i].min()))
        return norm_img.astype(np.uint8)

def blackhat(img, kernel_size=[200,200]):
    kernel = np.ones(kernel_size, np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return 255 - result

def median(img, size=21):
    median = cv2.medianBlur(img, size)
    return median

def low_pass_filter(img, d=15):
    fourier = np.fft.fft2(img) # fourier transfrom
    shifted =  np.fft.fftshift(fourier) # fftshift
    gaus = gaussian_filter(d, fourier.shape)
    filtered_fourier = shifted * gaus # apply filter
    fourier_result = np.fft.ifftshift(filtered_fourier) # inverse fftshift
    result = np.fft.ifft2(fourier_result) # inverse fourier transform
    return result.real.astype(np.uint8)

def low_pass_filter_rgb(img, d=15):
    result = np.zeros(img.shape)

    for i in range(img.shape[2]):
        result[:,:,i] = low_pass_filter(img[:,:,i], d)
    return result.clip(0, 255).astype(np.uint8)

def dilate(img, kernel_size=[5,5], iterations=1):
    # The hairs are darker than the image
    # so to 'erode' them, you have to dilate the image
    kernel = np.ones(kernel_size, np.uint8)
    result = cv2.dilate(img.astype(np.uint8), kernel, iterations = iterations)
    return result

def gaussian_filter(d, shape):
    center = [shape[0]//2, shape[1]//2]
    filt = np.fromfunction( lambda i, j: (np.exp(-(distance([i,j], center)**2) / (2*d**2))), shape) 
    return filt

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# Segmentation methods

def threshold(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return result

def watershed(img, label_seg=20):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = rank.median(img, disk(2))
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(denoised, disk(2))
    result = segmentation.watershed(gradient, markers)
    result[result > label_seg] = 255
    result[result <= label_seg] = 0
    return img_as_ubyte(result)

def mser(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = np.zeros(img.shape)
    seg = cv2.MSER_create()
    regions, _ = seg.detectRegions(img_as_ubyte(img))
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    for contour in hulls:
        cv2.fillPoly(result, [contour], 1, 255)
    return (result * 255).astype(np.uint8)

def k_means(img):
    pixel_values = img.reshape((-1, 3))
    pixel_values = pixel_values.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = centers[labels.flatten()].reshape(img.shape).clip(0, 255).astype(np.uint8)
    return threshold(segmented_image)