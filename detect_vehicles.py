# Vehicle Detection

from collections import deque
import glob
import multiprocessing
import os
import pickle
import time

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import pandas
from scipy.ndimage.measurements import label
import scipy.stats as stats
from scipy.stats import expon as sp_expon
from scipy.stats import randint as sp_randint
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm.classes import SVC

debug_image = False

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def slide_window(img):
    '''method to compute sliding windows over image for vehicle detection'''

    top = np.int(img.shape[0]/2)
    x_start_stop=[None, None]
    y_start_stop=[top, None]
    xy_window_min=(48,48)
    xy_window_max=(500,500)
    xy_overlap=(0.5, 0.5)
    
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
        
    # Initialize a list to append window positions to
    window_list_all = []
    
    for i in range(10):
        window_list = []
        xy_window = (xy_window_min[0] + np.int(i*(xy_window_max[0]-xy_window_min[0])/5), xy_window_min[1] + np.int(i*(xy_window_max[1]-xy_window_min[1])/5))

        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        
        if debug_image == True:
            plt.imshow(draw_boxes(img, window_list))
            plt.show()
            
        window_list_all.extend(window_list)
    # Return the list of windows
    return window_list_all


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):  
    '''method to extract features from a single image'''
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def single_image_features_tupled(args):
    img_name, flip, cropping, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat = args
    image = cv2.imread(img_name)
    if flip is True:
        image = cv2.flip(image, 1)
    if cropping is not None:
        xmin = min(cropping[0], cropping[1])
        xmax = max(cropping[0], cropping[1])
        ymin = min(cropping[2], cropping[3])
        ymax = max(cropping[2], cropping[3])

        if (xmin < xmax) and (ymin < ymax) and (ymin >= np.int(image.shape[0]/2)):
            image = cv2.resize(image[ymin:ymax, xmin:xmax], (64, 64))
        else:
            return None
    return single_img_features(image, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)


def extract_features(file_names, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''extract features from list of image names'''
    global multip
    regular = multip.map(single_image_features_tupled, [(file_name, False, None, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat) for file_name in file_names])
    flipped = multip.map(single_image_features_tupled, [(file_name, True, None, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat) for file_name in file_names])
    return np.concatenate((regular, flipped))
    #return regular
    
def extract_udacity_features(udacity_car_images, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''extract features from list of image names'''
    global multip
    return list(filter(lambda x: x is not None, multip.map(single_image_features_tupled, [(udacity_car_image[4], udacity_car_image[:4], color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat) for udacity_car_image in udacity_car_images])))

def search_windows_with_args(args):
    img, window, clf, scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat = args

    #3) Extract the test window from original image
    test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
    #4) Extract features for that window using single_img_features()
    features = single_img_features(test_img, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    #5) Scale extracted features to be fed to classifier
    test_features = scaler.transform(np.array(features).reshape(1, -1))
    #6) Predict using your classifier
    prediction = clf.predict(test_features)
    #7) If positive (prediction == 1) then save the window
    return window if prediction == 1 else None
        
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    '''method to find vehicles within precomputed search windows in an image'''
    global multip

    #1) Create an empty list to receive positive detection windows
    on_windows = multip.map(search_windows_with_args, [(img, window, clf, scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat) for window in windows])
    #2) Iterate over all windows in the list

    #8) Return windows for positive detections
    return list(filter(None, on_windows))

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''method to draw bounding boxes on image'''

    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(imcopy, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return imcopy

### TODO: Tweak these parameters and see how the results change.
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 3 # HOG cells per block
hog_channel = 2 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

def train_vehicle_classifier():
    '''train a classifier to identify vehicles in a single image'''
    global classifier
    global X_scaler

    # Read in cars and notcars
    cars = glob.glob('vehicles/KITTI_extracted/*.png')
    notcars = glob.glob('non-vehicles/Extras/*.png')
    
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
     
 #==============================================================================
 #    #now get cars from Udacity dataset
 #    udacity_labels = pandas.read_csv("object-detection-crowdai/labels.csv")
 #    udacity_car_labels = udacity_labels.loc[udacity_labels['Label'] == 'Car', ['xmin', 'xmax', 'ymin', 'ymax', 'Frame']] 
 #    # x and y labels are bad. swap them.
 #    udacity_car_labels[['xmax', 'ymin']] = udacity_car_labels[['ymin', 'xmax']]
 # 
 #    udacity_car_images = []
 #    for index, row in udacity_car_labels.sample(frac=0.1).iterrows():
 #        udacity_car_images.append((row["xmin"],row["xmax"],row["ymin"],row["ymax"],"object-detection-crowdai/%s" % row["Frame"]))
 #     
 #    udacity_car_features = extract_udacity_features(udacity_car_images, color_space=color_space, 
 #                            spatial_size=spatial_size, hist_bins=hist_bins, 
 #                            orient=orient, pix_per_cell=pix_per_cell, 
 #                            cell_per_block=cell_per_block, 
 #                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
 #                            hist_feat=hist_feat, hog_feat=hog_feat)
 #     
 #    all_car_features = np.concatenate((car_features, udacity_car_features))
 #==============================================================================

    all_car_features = car_features
    X = np.vstack((all_car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = RobustScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    print("cars: %s, noncars: %s" % (len(all_car_features), len(notcar_features)))
    
    # Define the labels vector
    y = np.hstack((np.ones(len(all_car_features)), np.zeros(len(notcar_features))))
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC with a random search over parameters
    
    params = {'svc__C': sp_expon(scale = 0.1), "rf__max_depth": [3, 5, 10, None],
              "rf__max_features": [5,10,25,'auto'],
              "rf__min_samples_split": sp_randint(2, 15),
              "rf__min_samples_leaf": sp_randint(1, 15)}    
    
    classifier = VotingClassifier(estimators=[('svc', LinearSVC()), ('rf', RandomForestClassifier())], voting='hard')
    
    classifier = RandomizedSearchCV(classifier, param_distributions=params, n_iter = 20, n_jobs = multiprocessing.cpu_count(), cv = 3, verbose = 1)
    
    
    # Check the training time for the SVC
    t=time.time()
    classifier.fit(X_train, y_train)
    t2 = time.time()
    print("best estimator score: %s with %s (%s)" % (classifier.best_score_, classifier.best_estimator_, classifier.best_params_))
    print(round(t2-t, 2), 'Seconds to train classifier...')
    # Check the score of the SVC
    print('Test Accuracy of classifier = ', round(classifier.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    
    with open('best_classifier', 'wb') as f:
        pickle.dump(classifier, f)

def reset_measurements():
    """reset vehicle state between videos / still images"""


def process_image(image):
    """completely process a single BGR image"""
    global classifier
    global X_scaler

    windows = slide_window(image)

    flagged_windows = search_windows(image, windows, classifier, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat) 
                      
    image_with_flagged_windows = draw_boxes(image, flagged_windows, color=(0, 0, 255), thick=6)
    
    if debug_image == True:
        plt.figure(figsize=(20, 10))
        plt.imshow(image_with_flagged_windows)
        plt.show()
    
    heat = np.zeros_like(image[:,:,0]).astype(np.float)    
    
    # Add heat to each box in box list
    heat = add_heat(heat, flagged_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    if debug_image == True:
        plt.figure(figsize=(20, 10))
        plt.imshow(heatmap, cmap='hot')
        plt.show()
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    final = draw_labeled_bboxes(image_with_flagged_windows, labels)

    if debug_image == True:
        plt.figure(figsize=(20, 10))
        plt.imshow(final)
        plt.show()

    return final

#===============================================================================
# 
# def make_noncar_boxes(img, patches, y_min = 583, y_max = 980, n_per_img = 6, mean_size = 96):
#     '''
#     Creates n_per_image random box coordinates per image, that don't contain a car 
#     (or more precisely, do not intersect with areas marked as containing a car).
#     '''
#     boxes = []
# 
#     # Make a blank image
#     canvas = np.zeros_like(img[:, :, 0])
#     car_boxes = patches.loc[:, 'xmin':'ymax']
#     img_copy = np.copy(img)
#     
#     # Make a map of the bounding boxes
#     for row in car_boxes.itertuples():
#         xmin = min(row[1], row[2])
#         xmax = max(row[1], row[2])
#         ymin = min(row[3], row[4])
#         ymax = max(row[3], row[4])
#         # Vertices
#         v0 = np.array([xmin, ymin])
#         v1 = np.array([xmax, ymin])
#         v2 = np.array([xmax, ymax])
#         v3 = np.array([xmin, ymax])
#         # Draw box on canvas
#         cv2.fillConvexPoly(canvas, np.array([v0, v1, v2, v3]), (255.))
#         tried, retained = 0, 0
# 
#         while len(boxes) < n_per_img:  # Try new random boxes until we have as many as specified
#             tried += 1
#             # Randomly build boxes
#             # Top left corner:
#             x0, y0 = np.random.randint(0, 1760), np.random.randint(y_min, y_max)
#             # Bottom right corner (size is taken from a truncated normal distribution)
#             lower, upper = 32, 160
#             sigma = mean_size / 3
#             box_size = int(stats.truncnorm((lower - mean_size) / sigma, 
#                 (upper - mean_size) / sigma, loc = mean_size, 
#                 scale = sigma).rvs())
#             x1, y1 = x0 + box_size, y0 + box_size
# 
#             # Extract this patch of the canvas
#             box_array = canvas[y0:y1, x0:x1]
# 
#             # Make sure this box doesn't intersect with car boxes already present on the canvas:
#             if (box_array == 0.).all() and (x1 <= 1920) and (y1 <= y_max):
#                 # In that case append to list
#                 boxes.append(((x0, y0), (x1, y1)))
#                 retained += 1
# 
#     return boxes  # Return list of non-car boxes for this image
# 
# def extract_patches(dataframe, min_x, n_lines = None, size = (64, 64)):
#     '''
#     Uses the CSV file provided with the Udacity data to extract picture patches of cars and resize
#     them to the specified size.
#     The dataframe passed as argument contains box coordinates, frame filenames and labels from
#     the images in the dataset.
#     '''
# 
#     car_imgs = []
#     noncar_imgs = []
# 
#     # Filter out all non-car patches
#     cars_only = dataframe.loc[dataframe['Label'] == 'Car', 
#         ['xmin', 'xmax', 'ymin', 'ymax', 'Frame']]  
#     # Warning: These column names are wrong! We need to swap two:
#     cars_only[['xmax', 'ymin']] = cars_only[['ymin', 'xmax']]
# 
#     # Sort the dataframe by 'Frame'
#     cars_only.sort_values('Frame', inplace = True)
# 
#     if n_lines:
#         cars_only = cars_only.iloc[:n_lines, :]
# 
#     for filename in cars_only['Frame'].unique():
#         img_patches = cars_only[cars_only['Frame'] == filename]
#         img = plt.imread("object-detection-crowdai/" + filename)
#         print("Processing file:", filename)
#         count = 0
#         for row in img_patches.itertuples():
#             # Warning: the column names are all scambled up in the CSV file
#             xmin = min(row[1], row[2])
#             xmax = max(row[1], row[2])
#             ymin = min(row[3], row[4])
#             ymax = max(row[3], row[4])
# 
#             if (xmin < xmax) and (ymin < ymax) and (xmin >= min_x):
#                 car_img_out = cv2.resize(img[ymin:ymax, xmin:xmax], size)
#                 car_imgs.append(car_img_out)
#                 plt.imshow(car_img_out)
#                 plt.show()
#                 count += 1
# 
#         if count != 0:
#             # Make a list of random non-car boxes
#             noncar_boxes = make_noncar_boxes(img, img_patches, y_min = 450, n_per_img = count)
#             # Extract these image patches and append them to noncar_imgs
#             for box in noncar_boxes:
#                 
#                 x0, y0 = box[0][0], box[0][1]
#                 x1, y1 = box[1][0], box[1][1]
#                 noncar_img_out = img[y0:y1, x0:x1]
#                 
#                 noncar_imgs.append(cv2.resize(noncar_img_out, (64, 64)))
# 
#     return car_imgs, noncar_imgs
#===============================================================================

with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    global multip
    multip = p

    print("Training vehicle classifier...")
    
    train_vehicle_classifier()
    
    # ENTRY POINT
    
    # run image processing on test images

    for test_image in glob.glob(os.path.join('test_images', '*.jpg')):
        print("Processing %s..." % test_image)
        reset_measurements()
        cv2.imwrite(os.path.join('output_images', os.path.basename(test_image)), cv2.cvtColor(
            process_image(cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB))
    
    # run image processing on test videos
    #===========================================================================
    # for file_name in glob.glob("*.mp4"):
    #     if "_processed" in file_name:
    #         continue
    #     print("Processing %s..." % file_name)
    #     reset_measurements()
    #     VideoFileClip(file_name).fl_image(process_image).write_videofile(
    #         os.path.splitext(file_name)[0] + "_processed.mp4", audio=False)
    #===========================================================================
