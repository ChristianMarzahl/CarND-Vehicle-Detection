import numpy as np
import cv2

# https://github.com/sumitbinnani/CarND-Vehicle-Detection
class ExtractCars(object):
    """description of class"""

    def __init__(self, feature_extractor, classifier, search_area=(400,650), windowSize=(64,64)):

       self.search_area = search_area
       self.feature_extractor = feature_extractor
       self.classifier = classifier
       self.windowSize = windowSize

       self.image_counter = 0

    def extract_cars(self, img):

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        sub_image = img[self.search_area[0]:self.search_area[1], :, :]
        sub_image_ori = sub_image.copy()
        sub_image = sub_image#.astype(np.float32)# / 255

        sub_image = self.feature_extractor.preprocess_image(sub_image)

        heatmap = np.zeros_like(sub_image[:,:,0]).astype(np.float)
   
        for layer, scale in self.pyramid(sub_image):

            stepSize = 18
            if scale == 1:
                stepSize = 32

            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in self.sliding_window(layer, stepSize=stepSize, windowSize=(self.windowSize[0], self.windowSize[1])):
                # if the current window does not meet our desired window size, ignore it
                if window.shape[0] != self.windowSize[1] or window.shape[1] != self.windowSize[0]:
                    continue

                if scale == 1 and y > 80 or scale == 2 and y > 32:
                    break

                features = self.feature_extractor.extract_features(window)              
                prediction = self.classifier.predict_probability(features.reshape(1, -1))

                heatmap[int(y*scale):int((y + self.windowSize[1])*scale),int(x*scale):int((x + self.windowSize[0])*scale)] += prediction[0][1]

                #if prediction[0][1] > 0.5:
                #    #cv2.rectangle(sub_image_ori, (int(x*scale),int(y*scale)), (int((x + self.windowSize[0])*scale), int((y + self.windowSize[1])*scale)), (0, 255, 0), 2)
                #    #clone = sub_image_ori.copy()
                #    heatmap[int(y*scale):int((y + self.windowSize[1])*scale),int(x*scale):int((x + self.windowSize[0])*scale)]
                #    #print (prediction[0][1])
                #else:
                #    clone = sub_image_ori.copy()
                #    #cv2.rectangle(clone, (int(x*scale),int(y*scale)), (int((x + self.windowSize[0])*scale), int((y + self.windowSize[1])*scale)), (0, 255, 0), 2)

        kernel = np.ones((5,5),np.uint8)
        _, thresh_image = cv2.threshold(heatmap.astype(np.uint8),3,255,cv2.THRESH_BINARY)
        thresh_image = cv2.dilate(thresh_image,kernel,iterations=5)
        im2, contours, hierarchy = cv2.findContours(thresh_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(sub_image_ori, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self.image_counter += 1
        cv2.imwrite("output_images/2/{0}.png".format(self.image_counter),sub_image_ori)

        sub_image_ori = cv2.cvtColor(sub_image_ori,cv2.COLOR_BGR2RGB)
        return sub_image_ori
        #cv2.imshow("Window", sub_image_ori)
        #cv2.waitKey(0)



    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def pyramid(self, image, scale_steps=[1,2]): # 1.5,2,2.5

        for scale in scale_steps:
            imshape = image.shape
            image = cv2.resize(image, (int(imshape[1] / scale), int(imshape[0] / scale)))
            # yield the next image in the pyramid
            yield image, scale