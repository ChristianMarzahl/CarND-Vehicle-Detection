import cv2
import numpy as np
import glob
import os
import pickle


from util.ExtractFeatures import ExtractFeatures
from util.SvmClassifier import SvmClassifier

# Divide up into cars and notcars
images = glob.glob('test_images/vehicles_smallset/*/*.jpeg', recursive=True)
cars = []
notcars = []

for image in images:
    if 'image' in image.split("\\")[-1] or 'extra' in image.split("\\")[-1]:
        notcars.append(image)
    else:
        cars.append(image)

# Feature extraction
extractFeatures = ExtractFeatures()
car_features = extractFeatures.extract_features_from_paths(cars)
not_car_features = extractFeatures.extract_features_from_paths(notcars)

#X = np.vstack((car_features, not_car_features)).astype(np.float64)
X = np.vstack((car_features, not_car_features)).astype(np.float64)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

# train classifier 
svm = SvmClassifier()
svm.train(X,y)

result = svm.predict(X)

# save classifier 

extracted_features_and_svm = {"svm":svm, "extractFeatures":extractFeatures}

with open('extracted_features_and_svm.pk', 'wb') as pickle_file:
    pickle.dump(extracted_features_and_svm, pickle_file)