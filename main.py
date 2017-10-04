import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import os
from moviepy.editor import VideoFileClip
from line import Line
from utils import *
from pipeline import *

images = glob.glob('test_images/*.jpg')
# images = glob.glob('test_images/*.png')
# images = glob.glob('test_images/test5.jpg')
for idx, fname in enumerate(images):
    image = mpimg.imread(fname)
    result = pipeline(image)

    left_line.detected = False
    left_line.recent_fits = []
    left_line.best_fit = None

    right_line.detected = False
    right_line.recent_fits = []
    right_line.best_fit = None

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result, cmap='gray')

    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def process_image(image):
    result = pipeline(image)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(result)
    # plt.show()

    return result

VIDEO_OUTPUT_DIR = 'test_videos_output/'
if not os.path.isdir(VIDEO_OUTPUT_DIR):
    os.mkdir(VIDEO_OUTPUT_DIR)

def process_video(video_input, video_output):
    clip = VideoFileClip(os.path.join(os.getcwd(), video_input))
    processed = clip.fl_image(process_image)
    processed.write_videofile(os.path.join(VIDEO_OUTPUT_DIR, video_output), audio=False)

process_video('project_video.mp4', 'project_video.mp4')
# process_video('challenge_video.mp4', 'challenge_video.mp4')
# process_video('harder_challenge_video.mp4', 'harder_challenge_video.mp4')
