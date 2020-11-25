import numpy as np
from cv2 import cv2

def get_TPR(TP,FN):
    return TP/(TP+FN)

def get_FPR(FP, TN):
    return FP/(FP+TN)

# B&W images aka 1s and 0s matrix
# returns TP, TN, FP FN
def compare_seg(real_seg, test_seg):
    TP = np.sum(cv2.bitwise_and(real_seg,test_seg))
    TN = np.sum(cv2.bitwise_not(cv2.bitwise_or(real_seg,test_seg)))
    FP = np.sum(cv2.bitwise_and(cv2.bitwise_not(real_seg),test_seg))
    FN = np.sum(cv2.bitwise_and(cv2.bitwise_not(test_seg),real_seg))
    TPR = get_TPR(TP, FN)
    FPR = get_FPR(FP, TN)
    return TPR, FPR, TP, TN, FP, FN