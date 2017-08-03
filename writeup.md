**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

As training data, I chose to use the KITTI vehicle images dataset and the and Extra non-vehicle images dataset provided, which includes positive and negative examples of vehicles.

Here is an example of a vehicle and "not vehicle":

![alt text][image1]

HOG features are extracted from each image using a method called [`single_img_features`](https://github.com/dinoboy197/CarND-Vehicle-Detection/blob/master/detect_vehicles.py#L68-L114). This method converts the color space of the image into the [YCrCb color space](https://en.wikipedia.org/wiki/YCbCr) (Luma, Blue-difference chroma and Red-difference chroma). Next, the color channels are separated and each channel is passed through a HOG gradient compute method, called [`get_hog_features`](https://github.com/dinoboy197/CarND-Vehicle-Detection/blob/master/detect_vehicles.py#L26-L45). This method computes the histogram of gradient features for the selected color channels.

Here is an example of color channels and their extracted HOG features using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

My general optimization strategy for this project was to increase the accuracy of the vehicles detected in video by tuning one feature extraction parameter at a time: the feature type (HOG, spatial bins, color histogram bins), color space, and various hyperparameters for the feature type selected. While not a complete grid search of all available parameters for tuning in the feature space, my results show reasonably good performance.

My first investigation was into the effect of HOG, color histogram, and spatial binned features separately. I found that HOG features alone led to the most robust classifier in terms of accuracy without much tuning; addition of either color histogram or spatial features greatly increased the number of false positive vehicle detections. For this project, I chose to focus solely on HOG features and optimize them. This project does not discount the possible usefulness of histogram or spatial features, I have simply chosen to focus my research in this project on HOG features.

I briefly experimented with different color spaces for HOG feature extraction. I quickly discarded RGB features, whose performance both in training and on the videos was subpar to the other spaces. Eventually, the YCrCb color space showed as particularly performant on both the training images and in video compared to the other color spaces investigated (YUV, LUV, HLS, HSV).

Next, I optimized various hyperparameters of the HOG transformation: number of HOG channels, number of HOG orientations, and pixels per cell (cells per block remained at 2 for all tests). In studying the classification results from both test images, test video and the project video, I found that the following parameters yielded the best classification accuracy:

* HOG channels: all
* Number of HOG orientations: 9
* Pixels per cell: 8

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM classifier for detecting vehicles in images by extracting features from a training set, scaling the feature vectors, and finally using the training the model in a method called [`train_vehicle_classifier`](https://github.com/dinoboy197/CarND-Vehicle-Detection/blob/master/detect_vehicles.py#L332-L390).

Each vehicle and non-vehicle image had HOG features (as described above) extracted. To increase the generality of my classifier, [each training image is flipped on the horizontal axis in the dataset](https://github.com/dinoboy197/CarND-Vehicle-Detection/blob/master/detect_vehicles.py#L132-L136), which increased the total size of the training data to 11932 vehicle images and 10136 non-vehicle images. The relative equality of the counts of vehicle and non-vehicle images reduces the bias of any classifier towards making vehicle or non-vehicle predictions. Each one dimensional feature vector was scaled using the [Scikit Learn `RobustScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html), which "scales features using statistics that are robust to outliers" by using the median and interquartile range, rather than the sample mean as the [`StandardScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) does.

After scaling, the feature vectors were split into a training and test set, with 20% of the data used for testing.

Finally, a binary SVM classifier was trained using a linear kernel (using the [SciKit Learn `LinearSVC` model](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)). Results based on the training data showed a 99.82% accuracy on the test data.

Upon completion of the training pipeline, I continued to experiment with other classifiers to determine if I could gain better classifier performance on the test set and in videos. To do so, I experimented with random forests using the [SciKit Learn `RandomForestClassifier` model](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), using a grid search over various parameters for optimization (using [SciKit Learn `GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), and final voting of classifier based on the [SciKit Learn `VotingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)). My results showed that the random forest classifier performed on-par with the support vector machine but required more hyperparameter tuning, and so the code remains with only the `LinearSVC`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After implementing a basic classifier with reasonable performance on the training data provided, I moved on to detecting vehicles in test images and video. I used a "sliding window" approach in which a window which encapsulates a sub-image of the full image is moved across the full image. Features are extracted from the sub-image, and the classifier determines if there is a vehicle present or not. The window slides both horizontally and vertically across the image. I chose a window size of 64x64 pixels, with an overlap of 75% as the detection window slides. Once all windows have been searched, the list of windows in which vehicles were detected was returned (which may include some overlap; this is dealt with by a heatmap rectification as described below). As an early optimization to eliminate extra false positive vehicle detections, I chose to limit the vertical span of searching from the just above the top of the horizon to just above the vehicle engine hood in the image (based on visual inspection).

As a speed optimization, the sliding window search computes HOG features for the entire image first, then the sliding windows pull in the HOG features captured by that window, and other features are computed for that window. Together with Python's multiprocessing library, the speed improvements enabled experimentation across the various parameters in a reasonable time (~15 minutes to process the entire project video).

![alt text][image3]

In an attempt to improve vehicle detection accuracy in the project video, I experimented with changing the window size and including window sizes of multiples of 32: 64, 96, 128, 160, and 192. Overall vehicle detection accuracy decreased when using any of the other sizes. Additionally, I tried to use multiple sizes at once; this caused problems further down in the vehicle detection pipeline (the bounding box smoother).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As mentioned in above sections, I manually performed a restricted grid search of the feature extraction types and parameters before settling on a group of features and classifier type which yielded good results. Here are some sample images showing the boxes around images in example which were classified as vehicles:

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

My pipeline successfully generates [a video stream which shows bounding boxes around the vehicles](project_video_processed.mp4). While the bounding boxes are somewhat wobbly, and there are some false positives, the vehicles in the driving direction are identifed with relatively high accuracy.

In this project, I spent a considerable amount of time adjusting the recall / precision tradeoff of vehicle detection in the final image, tweaking the heatmap threshold and pixels per cell of HOG extraction until I found a satisfactory video stream. As with many machine learning classification problems, as false negatives go down, false positives go up. The heatmap threshold could be adjusted up or down to suit the end use case.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The pipeline records the positions of positive detections in each frame of the video. Positive detection regions are tracked for the current and previous four frames at each frame processing. The five total positive detections are stacked together (each pixel inside a region is one count), and then the final stacked heatmap is thresholded identify vehicle positions (eleven counts or more pixel being used as the threshold). This mechanism exists within the [`process_image`](https://github.com/dinoboy197/CarND-Vehicle-Detection/blob/master/detect_vehicles.py#L426-L443) method. I then used [SciPy's `label`](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) to identify individual blobs in the heatmap. Each blob is assumed to correspond to a vehicle, and each blob was used to construct a vehicle bounding box which was drawn over the image frame.

Here is an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a frame and its corresponding heatmap:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][image6]


### Here the resulting bounding boxes are drawn the image:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project proved interesting as it required good judgement to determine where the easiest sections were to optimize vehicle detection at any given time. The most challenging part of the project was the search over the large number of parameters in the training and classification pipeline. Many different pieces could be adjusted, including:
* size and composition of the training image set
* choice of combination of features extracted (HOG, spatial, and color histogram)
* parameters for each type of feature extraction
* choice of machine learning model (SVC, random forest, etc)
* hyperparameters of machine learning model
* sliding window size and stride
* heatmap stack size and thresholding variable

Rather than completing an exhaustive grid search on all possibilities (which would not only have been computationally infeasible in Python but also likely to overfit the training data), I made incremental educated guesses about which choices to make and which parameters to tune. Overall, I was satisfied with the output but would like to make improvements.

Problems in the current implementation that could be improved upon include:
* reduction in number of false positive detections, in the form of:
  * small detections sprinkled around the video - could add more post-processing to filter out small boxes after final heat map label creation
  * a few large detections in shadow areas or with highway signs - continue the project with other spatial features?
* not detecting the entirety of the vehicle
  * often the side of the vehicles are missed - include more training data with side images of vehicles
  * side detections can be increased by lowering the heatmap masking threshold, at the expense of more false positive vehicle detections

The pipeline would likely fail to detect in various situations, including (but not limited to):
* vehicles other than cars - fix with more training data with other vehicles
* nighttime detection - fix with different training data and possibly different feature extraction types / parameters
* detection of vehicles driving perpandicular to vehicle - adjust heatmap queuing value and thresholding, possibly training data, too


