from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, json
from compare import compare_seg

def kernel(n):
    return np.ones((n,n),np.uint8)


# This function test a segmentation function and return its TPR and FPR
# segmentation_function must recieve an image and return B&W segmentation
# results_file_name will be the name of the file with the results
def test_all(
    segmentation_function, 
    results_file_name, 
    img_folder = "images/original", 
    seg_folder = "images/segmented"):
    results = {}
    avg_TPR = 0
    avg_FPR = 0
    avg_TP = 0
    avg_TN = 0
    avg_FP = 0
    avg_FN = 0
    cont = 0
    for img_name in sorted(os.listdir(img_folder)):
        cont += 1
        aux_index = img_name.index("i")
        n = img_name[0:aux_index]
        # Real image
        img_path = img_folder + "/" + img_name
        img = cv2.imread(img_path)

        # Segmented by function
        func_seg_img = segmentation_function(img)

        # Real segmentation
        real_seg_path = seg_folder + "/" + n + "seg.png"
        real_seg_img = cv2.imread(real_seg_path,0)

        # Compare
        TPR, FPR, TP, TN, FP, FN = compare_seg(real_seg_img, func_seg_img)
        this_res = {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "TPR": TPR,
            "FPR": FPR,
        }
        results[n] = this_res
        avg_TPR += TPR
        avg_FPR += FPR
        avg_TP += TP
        avg_TN += TN
        avg_FP += FP
        avg_FN += FN

    avg_TPR = avg_TPR/cont
    avg_FPR = avg_FPR/cont
    avg_TP = avg_TP/cont
    avg_TN = avg_TN/cont
    avg_FP = avg_FP/cont
    avg_FN = avg_FN/cont

    results["average_TPR"] = avg_TPR
    results["average_FPR"] = avg_FPR
    results["average_TP"] = avg_TP
    results["average_TN"] = avg_TN
    results["average_FP"] = avg_FP
    results["average_FN"] = avg_FN

    # Save results
    with open('results/{}.txt'.format(results_file_name), 'w') as outfile:
        json.dump(results, outfile)
        

# -> gray -> threshold -> open -> dilate(5) x3
# -> blur(9) -> dilate(7) x3 -> blur(11) -> close(15)
def simplest_method(img, show=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ ,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel(3))
    final = cv2.morphologyEx(opening, cv2.MORPH_DILATE, kernel(5),iterations=3)
    final = cv2.medianBlur(final,9)
    final = cv2.morphologyEx(final, cv2.MORPH_DILATE, kernel(7),iterations=3)
    final = cv2.medianBlur(final,11)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel(15))
    if show:
        cv2.imshow('Original image',img)
        cv2.imshow('Gray image', gray)
        cv2.imshow('Threshold image', thresh)
        cv2.imshow('Final image', final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return final


# Displays the results for the n image
def test_one_image(
    segmentation_function, 
    n,
    img_folder = "images/original", 
    seg_folder = "images/segmented"):
    n = str(n)
    # Real image
    img_name = "{}img.jpg".format(n)
    img_path = img_folder + "/" + img_name
    img = cv2.imread(img_path)

    # Segmented by function
    func_seg_img = segmentation_function(img)

    # Real segmentation
    real_seg_path = seg_folder + "/" + n + "seg.png"
    real_seg_img = cv2.imread(real_seg_path,0)

    # Compare
    TPR, FPR, TP, TN, FP, FN = compare_seg(real_seg_img, func_seg_img,show=True)
    return TPR, FPR, TP, TN, FP, FN

    
# img_path = "images/original/1img.jpg"
# img = cv2.imread(img_path)
# test = simplest_method(img, show=True)
# test = simplest_method(img, show=False)

# real_path = "images/segmented/1seg.png"
# real = cv2.imread(real_path,0)

# print(compare_seg(real,test,show=True))
# print(test_one_image(simplest_method,7))

# test_all(simplest_method,"simplest_method")
