import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
from tqdm import tqdm

class ExtractFeatures(object):
    """description of class"""

    def __init__(self, color_space='YCrCb', spatial_size=(32, 32), hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL', spatial_feat=True, hist_feat=True,
                     hog_feat=True, vis=False, feature_vec=True):

        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins  = hist_bins
        self.orient  = orient
        self.pix_per_cell  = pix_per_cell
        self.cell_per_block  = cell_per_block
        self.hog_channel  = hog_channel
        self.spatial_feat  = spatial_feat
        self.hist_feat  = hist_feat
        self.hog_feat  = hog_feat
        self.vis = vis
        self.feature_vec = feature_vec

    def extract_features_from_paths(self, image_paths):

        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in tqdm(image_paths):
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
 
            # Convert color space
            feature_image = self.preprocess_image(image)

            # extract features for the image
            features_image = self.extract_features(feature_image)

            features.append(features_image)
        # Return list of feature vectors
        return features

    def extract_features(self, image):
        file_features = []

        if self.spatial_feat == True:
            spatial_features = self.bin_spatial(image, self.spatial_size)
            file_features.append(spatial_features)
        if self.hist_feat == True:
            hist_features = self.color_hist(image, nbins=self.hist_bins)
            file_features.append(hist_features)
        if self.hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(image.shape[2]):
                    hog_channel_features = self.get_hog_features(image[:,:,channel], self.orient, self.pix_per_cell, self.cell_per_block, vis=self.vis, feature_vec=self.feature_vec)
                    hog_features.append(hog_channel_features)
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = self.get_hog_features(image[:,:,self.hog_channel], self.orient, self.pix_per_cell, self.cell_per_block, vis=self.vis, feature_vec=self.feature_vec)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        return np.concatenate(file_features).reshape(1, -1)

    def preprocess_image(self, image):
        
        # apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)    

        return feature_image


    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        if vis == True:
            # Use skimage.hog() to get both features and a visualization
            fd, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                        cells_per_block=(cell_per_block,cell_per_block), visualise=vis, feature_vector=feature_vec)
            features = fd 
            hog_image = hog_image 
            return features, hog_image
        else:      
            # Use skimage.hog() to get features only
            return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell), cells_per_block=(cell_per_block,cell_per_block), visualise=vis, feature_vector=feature_vec)

    # Define a function to compute binned color features  
    def bin_spatial(self, img, size=(32, 32)):
        features = cv2.resize(img, size).ravel() 
        return features

    # Define a function to compute color histogram features  
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features