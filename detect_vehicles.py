# Vehicle Detection

from collections import deque
import cv2
import glob
import numpy as np
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

debug_image = False


def reset_measurements():
    """reset vehicle state between videos / still images"""


def process_image(image):
    """completely process a single BGR image"""

    final = image

    if debug_image == True:
        plt.figure(figsize=(20, 10))
        plt.imshow(final)
        plt.show()

    return final

# ENTRY POINT

# run image processing on test images
for test_image in glob.glob(os.path.join('test_images', '*.jpg')):
    print("Processing %s..." % test_image)
    reset_measurements()
    cv2.imwrite(os.path.join('output_images', os.path.basename(test_image)), cv2.cvtColor(
        process_image(cv2.cvtColor(cv2.imread(test_image), cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB))

# run image processing on test videos
for file_name in glob.glob("*.mp4"):
    if "_processed" in file_name:
        continue
    print("Processing %s..." % file_name)
    reset_measurements()
    VideoFileClip(file_name).fl_image(process_image).write_videofile(
        os.path.splitext(file_name)[0] + "_processed.mp4", audio=False)
