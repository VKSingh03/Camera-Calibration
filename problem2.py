import numpy as np
import cv2
import matplotlib.pyplot as plt

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height) # computing dimensions as tuple.
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
    image_paths=['image1.jpg','image2.jpg','image3.jpg','image4.jpg','image5.jpg','image6.jpg','image7.jpg','image8.jpg','image9.jpg','image10.jpg','image11.jpg','image12.jpg','image13.jpg']
    pattern_size = (6, 9) # 6 columns* 9 rows
    square_size = 21.5 # in mm

    imgpoints = []
    objpoints = []

    object_points = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    object_points[:,:2] = square_size * np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for image in image_paths:
        img = cv2.imread(image)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img,(6,9),None)
        # print(corners)
        if(ret):
            cv2.drawChessboardCorners(img,(6,9),corners,ret)
            imgpoints.append(corners)
            objpoints.append(object_points)
            img_rescaled = rescaleFrame(img, scale=0.5)
            cv2.imshow('Frame',img_rescaled)
            cv2.waitKey(10)
            # plt.imshow(img)
            # plt.show()
            # exit(0)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        # print(imgpoints2)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
        print( "Error for image ",i ," : ",(error) )
    print("\n\n Mean error: ", mean_error/len(objpoints), "\n\n")

    print("K matrix: \n", camera_matrix, "\n\n")