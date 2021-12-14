from __future__ import print_function
import cv2
import numpy as np
import argparse
import os

#I'm using this camera matrix for Windows, it seems to vary quite a bit when I try to calculate it though
cameraMatrix =  np.array([[539.09173765,   0.,         312.30788275, 0],
                  [  0.,         503.30900949, 225.51890671, 0],
 [  0.,           0.,           1.        , 0]], dtype=np.float32)

A1 = np.array(    [[1, 0, 0], #set A1[0][2] to -w/2, and A1[1][2] to -h/2 of best image
                   [0, 1, 0],
                   [0, 0, 0],
                   [0, 0, 1]], dtype=np.float32)

bigRotMat = np.array([[1.0,0,0,0], [0,1.0,0,0], [0,0,1.0,0], [0,0,0,1.0]], np.float32)

parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
parser.add_argument('--input1', nargs='+', help='Paths to input images and folders containing images', default=['example_images']) #default='box.png')
#parser.add_argument('--input2', help='Path to input image 2.', default='box_in_scene.png')
#parser.add_argument('--do_not_mirror', help='Disable webcam mirroring')
args = parser.parse_args()
img1 = []
filenames = []
corrected_file_args = []
for filename in args.input1: #turn folders into filelists (remove subfolders)
    if os.path.isdir(filename):
        corrected_file_args += [os.path.join(filename, x) for x in os.listdir(filename)]
    else:
        corrected_file_args += filename
for filename in corrected_file_args:
    img = cv2.imread(filename)#, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img1 += [img]
        filenames += [os.path.basename(filename)]

assert len(filenames) != 0, "No image files"
#img2 = cv2.imread(cv2.samples.findFile(args.input2), cv2.IMREAD_GRAYSCALE)
if img1 is None:# or img2 is None:
    print('Could not open or find the images!')
    exit(0)
#-- Step 1: Detect the keypoints using ORB Detector, compute the descriptors (formerly SURF)
minHessian = 400
detector = cv2.ORB_create() #.create(hessianThreshold=minHessian)
cam = cv2.VideoCapture(0+cv2.CAP_DSHOW) #remove cv2.CAP_DSHOW if not on Windows
keypoints_images = []
descriptors_images = []
for i in range(len(img1)):
    keypoints, descriptors = detector.detectAndCompute(img1[i], None)
    keypoints_images += [keypoints]
    descriptors_images += [descriptors]
    print(filenames[i], len(keypoints))
while True:
    ret_val, img2 = cam.read()
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None) #get webcam frame descriptors
    if type(descriptors2)==float and np.isnan(descriptors2):
        continue
    #if not args.do_not_mirror:
    #    img2 = cv2.flip(img2, 1)
    #cv2.imshow('my webcam', img)
    best_set_of_good_matches = []
    best_image_index = 0
    for i, (keypoints1, descriptors1) in enumerate(zip(keypoints_images, descriptors_images)):
        #keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
        #-- Step 2: Matching descriptor vectors with a FLANN based matcher
        # Since SURF is a floating-point descriptor NORM_L2 is used
        #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED) #if sift or surf
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        descriptors1 = np.float32(descriptors1)
        descriptors2 = np.float32(descriptors2)

        knn_matches = flann.knnMatch(descriptors1,descriptors2,k=2) #I don't think ORB is supposed to use L2,
        #-- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.7
        good_matches = []
        for m,n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        if len(good_matches) > len(best_set_of_good_matches):
            best_image_index = i
            best_set_of_good_matches = good_matches
    #-- Draw matches
    if len(best_set_of_good_matches) > 0:
        img_matches = np.empty((max(img1[best_image_index].shape[0], img2.shape[0]), img1[best_image_index].shape[1]+img2.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(img1[best_image_index], keypoints_images[best_image_index], img2, keypoints2, best_set_of_good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        #solvepnp stuff
        matches_image_keypoints = [keypoints_images[best_image_index][best_set_of_good_matches[i].queryIdx] for i in range(len(best_set_of_good_matches))]
        matches_frame_keypoints = [keypoints2[best_set_of_good_matches[i].trainIdx] for i in range(len(best_set_of_good_matches))]
        list_points3d = np.array([[keypoint.pt[0], keypoint.pt[1], 0] for keypoint in matches_image_keypoints], dtype=np.float32) #painting keypoints (Z=0 because flat)
        list_points2d = np.array([keypoint.pt for keypoint in matches_frame_keypoints], dtype=np.float32) #camera frame 2d keypoints
        if len(list_points3d) >= 4:
            result, rvec, tvec, _ = cv2.solvePnPRansac( list_points3d, list_points2d, cameraMatrix[:,:3], None)#, rvec, tvec,
                                #useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                                #inliers, flags )
            rotMat = cv2.Rodrigues(rvec)[0]
            bigRotMat[:3, :3] = rotMat
            tMat = np.array(    [[1, 0, 0, tvec[0]],
                                 [0, 1, 0, tvec[1]],
                                 [0, 0, 1, tvec[2]],
                                 [0, 0, 0, 1]], dtype=np.float32)
            A1[0][2] = 0 #-img1[best_image_index].shape[1]/2 #originally set A1[0][2] to -w/2, and A1[1][2] to -h/2 of best image, but this causes it to be displaced
            A1[1][2] = 0 #-img1[best_image_index].shape[0]/2

            transfo = cameraMatrix @ (tMat @ (bigRotMat @ A1))
            destination = cv2.warpPerspective(img1[best_image_index], transfo, (img2.shape[1], img2.shape[0]), cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
            cv2.imshow('destination', destination)
            mask = cv2.warpPerspective(np.ones_like(img1[best_image_index], np.float32), transfo, (img2.shape[1], img2.shape[0]), cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
            image_overlaid = (destination * mask + img2 * (1.0-mask))/255.0
        else:
            image_overlaid = img2

        #-- Show detected matches
        cv2.imshow('Good Matches', img_matches)
        cv2.imshow('overlay', image_overlaid)
        #cv2.waitKey()
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()