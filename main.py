from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, json
from compare import compare_seg

# This function test a segmentation function and return its TPR and FPR
# segmentation_function must recieve an image and return B&W segmentation
# results_file_name will be the name of the file with the results
def test_segmentation(
    segmentation_function, 
    results_file_name, 
    img_folder = "images/original", 
    seg_folder = "images/segmented"):
    for img_name in sorted(os.listdir(img_folder)):
        n = img_name[0]
        # Real image
        img_path = img_folder + "/" + img_name
        img = cv2.imread(img_path)

        # Segmented by function
        func_seg_img = segmentation_function(img)

        # Real segmentation
        real_seg_path = seg_folder + "/" + n + "seg.png"
        real_seg_img = cv2.imread(real_seg_path)

        # Compare
        TPR, FPR, TP, TN, FP, FN = compare_seg(real_seg_img, func_seg_img)
        res = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "TPR": TPR,
            "FPR": FPR,
        }

        # Save results
        with open('results/{}.txt'.format(results_file_name), 'w') as outfile:
            json.dump(res, outfile)
        
        


