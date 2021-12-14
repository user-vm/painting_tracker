import numpy as np
import cv2
import glob
import os
import argparse
import platform

NUM_WEBCAM_IMAGES = 50
CHESSBOARD_SIZE = (7,9)

parser =argparse.ArgumentParser("Iterativeley do chessboard calibration on input image sequence")
parser.add_argument("--input_folder", help="Image input folder, annotated output images will be placed in an out/ subfolder of this folder. If not specified, the webcam input will be used, and the images placed in ./out/.") #default='/home/vmiu/2019/opencv_undistortion/vive2_calibration_test/jpgs_less/*.jpg')
parser.add_argument("--pattern", help='Calibration pattern used, "circles" or "chessboard" ((6,7) circle or ' + str(CHESSBOARD_SIZE) + ' chessboard)', choices=["circles","chessboard"], default="chessboard")

args = parser.parse_args()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
if args.pattern == "circles":
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:7].T.reshape(-1,2)
else:
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#images = glob.glob('images_iphone/front_camera/*.jpg')
if args.input_folder is not None:
    globName = args.input_folder
    images = glob.glob(globName)
    imgShape = None
else:
    cam = cv2.VideoCapture(0+cv2.CAP_DSHOW if platform.system() == 'Windows' else 0)
    images = []
    globName = '.'
    imgShape = None

num_good_images = 0
#for i,fname in enumerate(images):
i=-1 #because images is empty when using webcam
while i<len(images):
    #print(fname)
    if args.input_folder is not None:
        images = max(i,0)
        fname = images[i]
        img = cv2.imread(fname)
    else:
        ret_val, img = cam.read()
        fname = str(len(images)).zfill(4)+".jpg"
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgShape = gray.shape[::-1]

    # Find the chess board corners
    if args.pattern == "circles":
        ret, corners = cv2.findCirclesGrid(gray, (6,7),None)
    else:
        ret, corners = cv2.findChessboardCorners(gray, (7,9)) #was (6,9)

    blurriness = cv2.Laplacian(img, cv2.CV_64F).var()
    if blurriness < 150: #reject blurry images
        ret = False
    img_copy = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_copy, str(ret) + ", " + str(num_good_images+ret) + ", " + str(blurriness), (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('my webcam', img_copy)

    #print(corners)

    if ret == True:
        #print(ret, corners)
        print(fname)
        if args.input_folder is None:
            images += [fname]
            num_good_images += 1
    else:
        if args.input_folder is not None:
            images[i] = None
            #os.remove(fname)
        else:
            images += [None]

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #corners2 = corners
        imgpoints.append(corners2)

        #save webcam images if using webcam
        if args.input_folder is None:
            imgPath = os.path.join(os.path.dirname(globName),"out_clean",os.path.basename(fname))
            os.makedirs(os.path.dirname(imgPath), exist_ok=True)
            cv2.imwrite(imgPath,img)

        # Draw and display the corners
        if args.pattern == "circles":
            img = cv2.drawChessboardCorners(img, (6,7), corners2,ret)
        else:
            img = cv2.drawChessboardCorners(img, (7,9), corners2,ret)
        #cv2.imshow('img',img)
        imgPath = os.path.join(os.path.dirname(globName),"out",os.path.basename(fname))
        os.makedirs(os.path.dirname(imgPath), exist_ok=True)
        print(imgPath)
        cv2.imwrite(imgPath,img)
        #cv2.waitKey(1000)

    if cv2.waitKey(1) == 27 or num_good_images >= NUM_WEBCAM_IMAGES:
        break  # esc to quit

    if args.input_folder is not None:
        i+=1
    
images = [i for i in images if i is not None]
#images = glob.glob('images_iphone/front_camera/*.jpg')
#images = glob.glob(globName)

cv2.destroyAllWindows()

print("len(objpoints) ", len(objpoints))
print("objpoints.shape ", objpoints[0].shape)
print("len(imgpoints) ", len(imgpoints))
print("imgpoints.shape ", imgpoints[0].shape)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgShape,None,None)

print("cameraMatrix = ", mtx)
print("distCoeffs = ", dist)

#NOTE: this last bit doesn't do anything unless you actually delete the bad images, and use
tot_error = 0
num_objpoints_added = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, None) #dist) #use dist if you want to use distortion coefficients
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    if error>0.1:
        images[i] = ""
        #os.remove(images[i])
        print(images[i], error,"deleted\n")
    else:
        tot_error += error
        print(images[i], error,"\n")
        num_objpoints_added+=1

print("total error: ", tot_error/num_objpoints_added)

