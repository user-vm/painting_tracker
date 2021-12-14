import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
#using https://stackoverflow.com/a/7021473

parser = argparse.ArgumentParser(description='Test warping image to simulate 3D rotation')
parser.add_argument('--input', help='Path to input image', default="box.png")
parser.add_argument('--Rx', help='Rotation around X axis (degrees)', default=0, type=float)
parser.add_argument('--Ry', help='Rotation around Y axis (degrees)', default=0, type=float)
parser.add_argument('--Rz', help='Rotation around Z axis (degrees)', default=0, type=float)
parser.add_argument('--Tx', help='Rotation around X axis (degrees)', default=0, type=float)
parser.add_argument('--Ty', help='Rotation around Y axis (degrees)', default=0, type=float)
parser.add_argument('--Tz', help='Rotation around Z axis (degrees)', default=0, type=float)
parser.add_argument('--rotation_order', choices=['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'], default='XYZ')

args = parser.parse_args()

source = cv2.imread(args.input)#, cv.IMREAD_GRAYSCALE)
assert source is not None, "Image " + args.input + " could not be opened"
w = source.shape[1]
h = source.shape[0]

#change to radians
args.Rx *= np.pi / 180
args.Ry *= np.pi / 180
args.Rz *= np.pi / 180

# Projection 2D -> 3D matrix
A1 = np.array(    [[1, 0, -w/2],
          [0, 1, -h/2],
          [0, 0,    0],
          [0, 0,    1]], dtype=np.float32)

# Rotation matrices around the X axis
Rx = np.array(    [[1,          0,           0, 0],
         [0, np.cos(args.Rx), -np.sin(args.Rx), 0],
         [0, np.sin(args.Rx),  np.cos(args.Rx), 0],
         [0,          0,           0, 1]], dtype=np.float32)

# Rotation matrices around the Y axis #TODO:fix
Ry = np.array(    [[np.cos(args.Ry),          0,           np.sin(args.Ry), 0],
                  [0, 1, 0, 0],
                  [-np.sin(args.Ry), 0,  np.cos(args.Ry), 0],
                  [0,          0,           0, 1]], dtype=np.float32)

# Rotation matrices around the Z axis #TODO:fix
Rz = np.array(    [[np.cos(args.Rz),          -np.sin(args.Rz),           0, 0],
                   [np.sin(args.Rz), np.cos(args.Rz), 0, 0],
                   [0, 0,  1, 0],
                   [0,          0,           0, 1]], dtype=np.float32)

Rdict = {"X":Rx, "Y":Ry, "Z":Rz}

# Translation matrix on the Z axis
T = np.array(    [[1, 0, 0, args.Tx],
         [0, 1, 0, args.Ty],
         [0, 0, 1, args.Tz],
         [0, 0, 0, 1]], dtype=np.float32)

f = 500

# Camera Intrisecs matrix 3D -> 2D
A2 = np.array([[f, 0, w/2, 0],
          [0, f, h/2, 0],
          [0, 0,   1, 0]])

transfo = A2 @ (T @ (Rdict[args.rotation_order[0]] @ Rdict[args.rotation_order[1]] @ Rdict[args.rotation_order[2]] @ A1))
unit = np.array([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]], dtype=np.float32)

destination = cv2.warpPerspective(source, transfo, (source.shape[1], source.shape[0]), cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

plt.imshow(destination)
plt.show()