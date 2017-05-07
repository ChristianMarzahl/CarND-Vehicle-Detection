import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.image as mpimg

from util.CamCalibration import  CameraCalibration
from util.ExtractFeatures import ExtractFeatures
from util.SvmClassifier import SvmClassifier
from util.ExtractCars import ExtractCars

cam_calibration = None
calibration_file = "camera_calibration.pickle"

cam_calibration = CameraCalibration(calibration_file)

with open('extracted_features_and_svm.pk', 'rb') as pickle_file:
    extracted_features_and_svm = pickle.load(pickle_file)

svm = extracted_features_and_svm["svm"]
extractFeatures = extracted_features_and_svm["extractFeatures"]

extractCars = ExtractCars(extractFeatures,svm)

image = mpimg.imread("test_images/test1.jpg")

extractCars.extract_cars(image)