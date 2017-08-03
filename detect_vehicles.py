# Vehicle Detection

from collections import deque
import glob
import itertools
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
from scipy.stats import randint as sp_randint
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler

debug_image = False


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    '''Computes Histogram Of Gradient (HOG) features for a given image'''
    # if vis = True, return features and HOG image
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(
                                      cell_per_block, cell_per_block),
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
    '''Computes binned spatial features for a given image'''
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    '''Computes histogram of color features for a given image'''
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    '''method to extract features from a single image'''
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
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
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def single_image_features_tupled(args):
    ''' computes single image features during training'''
    img_name, flip, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat = args
    image = cv2.imread(img_name)
    if flip is True:
        image = cv2.flip(image, 1)
    return single_img_features(image, color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)


def extract_features(file_names, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    '''extract features from list of image names in parallel'''
    global multip
    regular = multip.map(single_image_features_tupled, [(file_name, False, color_space, spatial_size, hist_bins,
                                                         orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat) for file_name in file_names])
    flipped = multip.map(single_image_features_tupled, [(file_name, True, color_space, spatial_size, hist_bins,
                                                         orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat) for file_name in file_names])
    return np.concatenate((regular, flipped))


def find_cars(args):
    '''method to find all vehicle bounding boxes given a single region to search'''
    xb, yb, cells_per_step, hog1, hog2, hog3, nblocks_per_window, ctrans_tosearch, window, scale, ystart, pix_per_cell, X_scaler, classifier, hog_channel, spatial_feat, spatial_size, hist_feat, hist_bins = args
    ypos = yb * cells_per_step
    xpos = xb * cells_per_step
    # Extract HOG for this patch
    final = np.array([])
    if (hog_channel == 0):
        final = np.hstack(
            (final, hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()))
    elif (hog_channel == 1):
        final = np.hstack(
            (final, hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()))
    elif (hog_channel == 2):
        final = np.hstack(
            (final, hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()))
    else:
        final = np.hstack(
            (final, hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()))
        final = np.hstack(
            (final, hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()))
        final = np.hstack(
            (final, hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()))

    xleft = xpos * pix_per_cell
    ytop = ypos * pix_per_cell

    # Extract the image patch
    subimg = cv2.resize(
        ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

    # Get color features
    if (spatial_feat == True):
        final = np.hstack((final, bin_spatial(subimg, size=spatial_size)))
    if (hist_feat == True):
        final = np.hstack((final, color_hist(subimg, nbins=hist_bins)))

    # Scale features and make a prediction
    reshaped = final.reshape(1, -1)
    test_features = X_scaler.transform(reshaped)
    #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
    test_prediction = classifier.predict(test_features)

    xbox_left = np.int(xleft * scale)
    ytop_draw = np.int(ytop * scale)
    win_draw = np.int(window * scale)
    this_window = ((xbox_left, ytop_draw + ystart),
                   (xbox_left + win_draw, ytop_draw + win_draw + ystart))
    found_car = test_prediction == 1

    return (this_window, found_car)


def find_cars_wrapper(img, scale, classifier, X_scaler, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                      cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    '''method to find vehicle detections in parallel'''
    global multip

    ystart = 400
    ystop = 656

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    if color_space != 'BGR':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YCrCb)
    else:
        ctrans_tosearch = np.copy(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(
            ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    
    if debug_image == False:
        hog1 = get_hog_features(
            ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(
            ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(
            ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if debug_image == True:
        hog1, vis1 = get_hog_features(
            ch1, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
        hog2, vis2 = get_hog_features(
            ch2, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
        hog3, vis3 = get_hog_features(
            ch3, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(20, 10))
        ax1.imshow(ch1, cmap='gray')
        ax1.set_title('Channel 1')
        ax2.imshow(vis1, cmap='gray')
        ax2.set_title('Channel 1 HOG')
        ax3.imshow(ch2, cmap='gray')
        ax3.set_title('Channel 2')
        ax4.imshow(vis2, cmap='gray')
        ax4.set_title('Channel 2 HOG')
        ax5.imshow(ch3, cmap='gray')
        ax5.set_title('Channel 3')
        ax6.imshow(vis3, cmap='gray')
        ax6.set_title('Channel 3 HOG')
        plt.show()

    all_windows = []
    matched_windows = []

    iter = itertools.product(range(nxsteps), range(nysteps))
    results = list(multip.map(find_cars, [(x[0], x[1], cells_per_step, hog1, hog2, hog3, nblocks_per_window, ctrans_tosearch, window,
                                           scale, ystart, pix_per_cell, X_scaler, classifier, hog_channel, spatial_feat, spatial_size, hist_feat, hist_bins) for x in iter]))
    all_windows = list(map(lambda x: x[0], results))
    matched_windows = list(filter(
        lambda x: x != None, map(lambda x: x[0] if x[1] == True else None, results)))

    if debug_image == True:
        plt.imshow(draw_boxes(img, all_windows, color=(0, 0, 255), thick=6))
        plt.show()

    return matched_windows


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
    '''Add individual heat components to a heatmap given bounding boxes'''
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    '''Zero out pixels below the threshold'''
    return np.maximum(heatmap - threshold, 0)


def draw_labeled_bboxes(img, labels):
    '''Draw labeled boxes on an image'''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox)-10, np.min(nonzeroy)),
                (np.max(nonzerox)-10, np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(imcopy, bbox[0], bbox[1], (0, 255, 0), 6)
    # Return the image
    return imcopy


def train_vehicle_classifier(identifier, color_space, hog_channel, hist_bins, spatial_size, orient, pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat):
    '''train a classifier to identify vehicles in a single image'''
    global classifier
    global X_scaler

    # Read in KITTI vehicles and non-vehicles
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

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = RobustScaler().fit(X)

    # Apply the scaler to the feature vectors
    scaled_X = X_scaler.transform(X)

    print("cars: %s, noncars: %s" % (len(car_features), len(notcar_features)))

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # train a support vector machine for classification based on the decision
    # boundary, with a linear kernel
    classifier = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    classifier.fit(X_train, y_train)
    t2 = time.time()
    #print("best estimator score: %s with %s (%s)" % (classifier.best_score_, classifier.best_estimator_, classifier.best_params_))
    print(round(t2 - t, 2), 'Seconds to train classifier...')
    # Check the score of the SVC
    print('Test Accuracy of classifier = ', round(
        classifier.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()


def reset_measurements():
    """reset vehicle state between videos / still images"""
    global last_heat_measurements

    last_heat_measurements = deque()


def process_image(image, color_space, hog_channel, hist_bins, spatial_size, orient, pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat, heat_threshold, detection_scales):
    """completely process a single BGR image"""
    global classifier
    global X_scaler
    global last_heat_measurements
    global debug_image

    # find all windows with vehicles identified
    flagged_windows_all = map(lambda x: find_cars_wrapper(image, x, classifier, X_scaler, color_space=color_space,
                                                          spatial_size=spatial_size, hist_bins=hist_bins,
                                                          orient=orient, pix_per_cell=pix_per_cell,
                                                          cell_per_block=cell_per_block,
                                                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                          hist_feat=hist_feat, hog_feat=hog_feat), detection_scales)
    flagged_windows = []
    for x in flagged_windows_all:
        flagged_windows.extend(x)

    if debug_image == True:
        # draw flagged windows on image
        image_with_flagged_windows = draw_boxes(
            cv2.cvtColor(image,cv2.COLOR_BGR2RGB), flagged_windows, color=(0, 0, 255), thick=6)
        plt.figure(figsize=(20, 10))
        plt.imshow(image_with_flagged_windows)
        plt.show()

    # initialize heat map
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, flagged_windows)

    # store last three bounding box measurements
    if len(last_heat_measurements) >= 5:
        last_heat_measurements.popleft()

    last_heat_measurements.append(heat)
    total_heat = np.sum(last_heat_measurements, axis=0)

    # Apply threshold to help remove false positives
    total_heat = apply_threshold(total_heat, heat_threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(total_heat, 0, 255)

    if debug_image == True:
        # draw heatmap
        f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(20, 10))
        ax1.imshow(cv2.cvtColor(draw_boxes(
            image, flagged_windows, color=(0, 0, 255), thick=6), cv2.COLOR_BGR2RGB))
        ax1.set_title('Image with vehicle detections')
        ax2.imshow(heatmap, cmap='hot')
        ax2.set_title('Heatmap')
        plt.show()

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    if debug_image == True:
        plt.imshow(labels[0], cmap='gray')
        plt.title('%s vehicles found' % labels[1])
        plt.show()

    # draw final bounding boxes on image
    final = draw_labeled_bboxes(image, labels)

    if debug_image == True:
        plt.figure(figsize=(20, 10))
        plt.imshow(final)
        plt.show()

    return final


def run():
    '''main training and video processing pipeline'''

    # Was ['YCrCb','RGB','YUV','HLS','HSV','LUV'] during experimentation
    color_space_options = ['YCrCb']
    orient = 9  # HOG orientations
    # HOG pixels per cell; was [6,7,8,9,10,11,12,13,14,15,16] during
    # experimentation
    pix_per_cell_options = [8]
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    # Spatial features on or off; was [False,True] during experimentation
    spatial_feat_options = [False]
    # Histogram features on or off; was [False,True] during experimentation
    hist_feat_options = [False]
    # HOG features on or off; was [False; True] during experimentation
    hog_feat_options = [True]
    # detection scales for sliding windows
    detection_scales = [1.0] # included combinations and permutations of 1.0,1.5,2.0,2.5 during experimentation

    classifier_arg_groups = itertools.product(
        color_space_options, pix_per_cell_options, spatial_feat_options, hist_feat_options, hog_feat_options)
    for classifier_args in classifier_arg_groups:
        # Python multiprocessing allows for use of full CPU resources
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            global multip
            global classifier
            global X_scaler
            try:
                del multip
            except:
                pass
            try:
                del classifier
            except:
                pass
            try:
                del X_scaler
            except:
                pass
            multip = p

            color_space, pix_per_cell, spatial_feat, hist_feat, hog_feat = classifier_args
            identifier = ",".join(map(str, classifier_args))

            if (spatial_feat or hist_feat or hog_feat) is False:
                # must have some features to compute!
                continue

            print("Training vehicle classifier %s..." % identifier)

            train_vehicle_classifier(identifier, color_space, hog_channel, hist_bins, spatial_size,
                                     orient, pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat)

            # run image processing on test images
            # was [5,6,7,8,9,10,11,12] during experimentation
            for heat_threshold in [11]:
                for test_image in glob.glob(os.path.join('test_images', '*.jpg')):
                    print("Processing %s..." % test_image)
                    reset_measurements()
                    cv2.imwrite(os.path.join('output_images', os.path.basename(test_image)), cv2.cvtColor(
                        process_image(cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_RGB2BGR), color_space, hog_channel, hist_bins, spatial_size, orient, pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat, heat_threshold, detection_scales), cv2.COLOR_BGR2RGB))

                # run image processing on test videos
                for file_name in glob.glob("*_video.mp4"):
                    if "_processed" in file_name:
                        continue
                    print("Processing %s..." % file_name)
                    reset_measurements()
                    VideoFileClip(file_name).fl_image(lambda x: process_image(x, color_space, hog_channel, hist_bins, spatial_size, orient, pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat, heat_threshold, detection_scales)).write_videofile(
                        os.path.splitext(file_name)[0] + "_processed_test.mp4", audio=False)

run()
