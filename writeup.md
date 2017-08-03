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
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

My general optimization strategy for this project was to increase the accuracy of the classifier by tuning one feature extraction parameter at a time among the feature, color space, and various hyperparameters for the feature set type selected. While not a complete grid search of all available parameters for tuning in the feature space, my results show reasonably good performance.

My first investigation was into the effect of HOG, color histogram, and spatial binned features. Using reasonable defaults provided by the sample code provided, HOG features alone led to the most robust classifier in terms of accuracy without much tuning; addition of either color histogram or spatial features greatly increased the number of false positive vehicle detections. For this project, I chose to focus on HOG features alone and optimize them. This doesn't discount the possible usefulness of histogram or spatial features, I've simply chosen to focus my research in this project on HOG features.

I briefly experimented with different color spaces for HOG feature extraction. I quickly discarded RGB features, whose performance both in training and on the videos was subpar to the other spaces. Eventually, the YCrCb color space showed as particularly well performing on both the training images and in video.

Next, I optimized various hyperparameters of the HOG transformation: number of HOG channels, number of HOG orientations, and pixels per cell (cells per block remained at 2 for all tests). In studying the classification results from both test images, test video and the project video, I found that the following parameters yielded the best classification accuracy:

* HOG channels: all
* Number of HOG orientations: 9
* Pixels per cell: 8

Some images which show classification

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM classifier for detecting vehicles in images by extracting features from a training set, scaling the feature vectors, and finally using the `fit` method of the `LinearSVC` classifier.

As training data, I chose to use the KITTI vehicle images dataset and the and Extra non-vehicle images dataset provided, which includes positive and negative examples of vehicles. To increase the generality of my classifier, I flipped each image on the horizontal axis, which increased the total size of the training data to 11932 vehicle images and 10136 non-vehicle images. The relative equality of the counts of vehicle and non-vehicle images reduces the bias of any classifier towards making vehicle or non-vehicle predictions.

Each vehicle and non-vehicle image had HOG features (as described above) extracted. Each one dimensional feature vector was scaled using a the [Scikit Learn `RobustScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html), which "scales features using statistics that are robust to outliers" by using the median and interquartile range, rather than the sample mean as the [`StandardScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) does.

After scaling, the feature vectors were split into a training and validation set, with 20% of the data used for testing.

Finally, a binary SVM classifier was trained using a linear kernel (using the [SciKit Learn `LinearSVC` model](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). Results based on the training data showed a 99.82% accuracy on the test data.

Upon completion of the training pipeline, I continued to experiment with other classifiers to determine if I could gain better classifier performance on the test set and in videos. To do so, I experimented with random forests using the [SciKit Learn `RandomForestClassifier` model](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), using a grid search over various parameters for optimization (using [SciKit Learn `GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), and final voting of classifier based on the [SciKit Learn `VotingClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html). My results showed that the random forest classifier performed on-par with the support vector machine, and so the code remains with only the `LinearSVC`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After implementing a basic classifier with reasonable performance on the training data provided, I moved on to detecting vehicles in test images and video. I used a "sliding window" approach in which a window which encapsulates a sub-image of the full image is moved across the image, then features are extracted from the sub-image, then the classifier determines if there is a vehicle or not in each sub-image. The window slides both horizontally and vertically across the image. I chose a window size of 64x64 pixels, with an overlap of 75% as the window slides around the image. Once all windows have been searched, the list of windows in which vehicles were detected was returned.

As a speed optimization, the sliding window search computes HOG features for the entire image first, then the sliding windows pull in the HOG features captured by that window, and other features are would be computed for that window in its entirety. Together with Python's multiprocessing library, the speed improvements enabled experimentation across the various parameters in a reasonable time.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

My classifier accuracy and video processing performance optimization is discussed above. Here are some sample images showing the boxes around images in example which were classified as vehicles:


![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

My pipeline successfully generates [a video stream which shows bounding boxes around the vehicles](project_video_processed.mp4). While the bounding boxes are somewhat wobbly, the vehicles in the driving direction are identifed with relatively high recall and precision.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

