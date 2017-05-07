import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test_images/bbox-example-image.jpg')

templist = ['test_images/cars/cutout1.jpg', 'test_images/cars/cutout2.jpg', 'test_images/cars/cutout3.jpg',
            'test_images/cars/cutout4.jpg', 'test_images/cars/cutout5.jpg', 'test_images/cars/cutout6.jpg']

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for box in bboxes:
        draw_img = cv2.rectangle(draw_img, box[0], box[1], color, thick)

    return draw_img # Change this line to return image copy with boxes

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes 
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    for template_path  in template_list:
        template = cv2.imread(template_path)
        # Read in templates one by one
        # Use cv2.matchTemplate() to search the image
        method  = eval(methods[1])
        res = cv2.matchTemplate(img,template, method)

        # Use cv2.minMaxLoc() to extract the location of the best match
        # Determine bounding box corners for the match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # Determine a bounding box for the match

        w, h = (template.shape[1], template.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        bbox_list.append((top_left, bottom_right))
    # Return the list of bounding boxes
    return bbox_list


# Add bounding boxes in this format, these are just example coordinates.
#bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
#          ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]

bboxes = find_matches(image,templist)

result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()
