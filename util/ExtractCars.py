import numpy as np
import cv2


class ExtractCars(object):
    """description of class"""

    def __init__(self, feature_extractor, classifier, search_area=(400,650), windowSize=(64,64)):

       self.search_area = search_area
       self.feature_extractor = feature_extractor
       self.classifier = classifier
       self.windowSize = windowSize

    def extract_cars(self, img):

        sub_image = img[self.search_area[0]:self.search_area[1], :, :]
        sub_image = sub_image.astype(np.float32) / 255

        sub_image_ori = sub_image.copy()
        sub_image = self.feature_extractor.preprocess_image(sub_image)

        for layer, scale in self.pyramid(sub_image):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in self.sliding_window(layer, stepSize=32, windowSize=(self.windowSize[0], self.windowSize[1])):
                # if the current window does not meet our desired window size, ignore it
                if window.shape[0] != self.windowSize[1] or window.shape[1] != self.windowSize[0]:
                    continue

                features = self.feature_extractor.extract_features(window)                

                prediction = self.classifier.predict(features)

                if prediction == 1:
                    cv2.rectangle(sub_image_ori, (int(x*scale),int(y*scale)), (int((x + self.windowSize[0])*scale), int((y + self.windowSize[1])*scale)), (0, 255, 0), 2)
                    clone = sub_image_ori.copy()
                else:
                    clone = sub_image_ori.copy()
                    cv2.rectangle(clone, (int(x*scale),int(y*scale)), (int((x + self.windowSize[0])*scale), int((y + self.windowSize[1])*scale)), (0, 255, 0), 2)

                
                cv2.imshow("Window", clone)
                cv2.waitKey(50)



    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def pyramid(self, image, scale_steps=[1.5,2,2.5]):
        # yield the original image
        yield image, 1

        for scale in scale_steps:
            imshape = image.shape
            image = cv2.resize(image, (int(imshape[1] / scale), int(imshape[0] / scale)))
            # yield the next image in the pyramid
            yield image, scale