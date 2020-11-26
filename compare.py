import numpy as np
from cv2 import cv2

def get_TPR(TP,FN):
    return TP/(TP+FN)

def get_FPR(FP, TN):
    return FP/(FP+TN)

# B&W images aka 255s and 0s matrix
# returns TP, TN, FP FN
def compare_seg(real_seg, test_seg, show=False):
    TP_img = cv2.bitwise_and(real_seg,test_seg)
    TN_img = cv2.bitwise_not(cv2.bitwise_or(real_seg,test_seg))
    FP_img = cv2.bitwise_and(cv2.bitwise_not(real_seg),test_seg)
    FN_img = cv2.bitwise_and(cv2.bitwise_not(test_seg),real_seg)
    if show:
        cv2.imshow('Real segmentation',real_seg)
        cv2.imshow('Test segmentation',test_seg)
        cv2.imshow('True Positive',TP_img)
        cv2.imshow('True Negative',TN_img)
        cv2.imshow('False Positive',FP_img)
        cv2.imshow('False negative',FN_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    TP = np.sum(TP_img)/255
    TN = np.sum(TN_img)/255
    FP = np.sum(FP_img)/255
    FN = np.sum(FN_img)/255
    TPR = get_TPR(TP, FN)
    FPR = get_FPR(FP, TN)
    return TPR, FPR, TP, TN, FP, FN