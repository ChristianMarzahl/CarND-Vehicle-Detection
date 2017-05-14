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

#image = mpimg.imread("test_images/test1.jpg")
image = cv2.imread("test_images/test1.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
sub_image_ori = extractCars.extract_cars(image)
#image = cv2.cvtColor(sub_image_ori,cv2.COLOR_RGB2BGR)
#cv2.imshow("Window", sub_image_ori)
#cv2.waitKey(0)

from moviepy.editor import VideoFileClip
video_output1 = 'project_video_output_rect.mp4'
video_input1 = VideoFileClip('project_video.mp4')
processed_video = video_input1.fl_image(extractCars.extract_cars)
processed_video.write_videofile(video_output1, audio=False)