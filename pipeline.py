import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import os
from moviepy.editor import VideoFileClip
from utils import *
from line import Line

CALIBRATION_FILE_DEFAULT = 'camera_cal/wide_dist_pickle.p'
dist_pickle = pickle.load(open(CALIBRATION_FILE_DEFAULT, "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

left_line = Line()
right_line = Line()

#loop over each image and apply preprocessing pipeline
def pipeline(image):

    undistorted = cal_undistort(image, mtx, dist)
    color_binary = color_thresh(undistorted)
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    gradx = abs_sobel_thresh(undistorted, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(undistorted, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(undistorted, sobel_kernel=ksize, thresh=(30, 100))
    dir_binary = dir_threshold(undistorted, sobel_kernel=ksize, thresh=(0.7, 1.3))

    # combined_binary = np.zeros_like(dir_binary)
    combined_binary = np.zeros_like(color_binary)
    combined_binary[(color_binary == 1) | ((gradx == 1) & (grady == 1)) \
                    | ((mag_binary == 1) & (dir_binary == 1))] = 1

    src = np.float32([[(200, 720), (453, 547), (835, 547), (1100, 720)]])
    dst = np.float32([[(320, 720), (320, 590), (960, 590), (960, 720)]])
    binary_warped, M, Minv = warp(combined_binary, src, dst)

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    if left_line.detected == False or right_line.detected == False:
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            #               (0,255,0), 2)
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            #               (0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        left_line.recent_fits.append(left_fit)
        right_line.recent_fits.append(right_fit)

        left_line.detected = True
        right_line.detected = True
    else:

        # left_fit = left_line.best_fit
        # right_fit = right_line.best_fit
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                             left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                               right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        left_line.recent_fits.append(left_fit)
        right_line.recent_fits.append(right_fit)


    if len(left_line.recent_fits) > left_line.n_fits:
        left_line.recent_fits.pop(0)

    if len(right_line.recent_fits) > right_line.n_fits:
        right_line.recent_fits.pop(0)

    left_line.best_fit = np.mean(left_line.recent_fits, axis=0)
    right_line.best_fit = np.mean(right_line.recent_fits, axis=0)

    left_fit = left_line.best_fit
    right_fit = right_line.best_fit

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    # result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    # print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    left_base_pos = (leftx_base - midpoint)*xm_per_pix
    right_base_pos = (rightx_base - midpoint)*xm_per_pix
    offset = (right_base_pos + left_base_pos)/2.0

    # text = 'Radius: ' + '{:4.0f}'.format(np.mean([left_curverad, right_curverad])) + 'm'
    # font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(result, text, (80, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # text = 'Offset: ' + '{:2.2f}'.format(offset) + 'm'
    # font = cv2.FONT_HERSHEY_DUPLEX
    # cv2.putText(result, text, (400, 80), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if offset < 0:
        str_side = 'left'
    else:
        str_side = 'right'

    cv2.putText(result,
                'Radius of Curvature = {:.0f}m'.format(np.mean([left_curverad, right_curverad])),
                (100, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1,
                color=(255,255,255),
                thickness=2,)
    cv2.putText(result,
                'Vehicle is {:.2f}m {:s} of center'.format(np.absolute(offset), str_side),
                (100, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1,
                color=(255,255,255),
                thickness=2,)

    return result
