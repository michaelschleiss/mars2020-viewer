
import pdr
import numpy as np
import glob
import cv2
from datetime import datetime
import re
import matplotlib.pyplot as plt

# Function to parse the .lbl file content
def parse_lbl_content(lbl_content):
    metadata = {}
    # Use regex to find key-value pairs
    items = re.findall(r'(\w+)=([^=]+?)(?=\s+\w+=|$)', lbl_content)
    for key, value in items:
        metadata[key.strip()] = value.strip().strip("'")
    return metadata

datetime_format = "%Y-%m-%dT%H:%M:%S.%f"
root = 'res/raw/'
filenames = sorted(glob.glob(root + '*.IMG'))

print(f"Number of images: {len(filenames)}")

data = pdr.read(filenames[0])
metadata = parse_lbl_content(data['IMAGE_HEADER'])
initial_time = datetime.strptime(metadata['START_TIME'], datetime_format)
initial_position = np.array(metadata['ORIGIN_OFFSET_VECTOR'].strip('()').split(','), dtype=float)

img2 = cv2.imread('res/JEZ_hirise_5m.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

kp2, des2 = sift.detectAndCompute(img2, None)

for i in range(10, len(filenames), 2):
    data = pdr.read(filenames[i])
    img1 = data['IMAGE']
    kp1, des1 = sift.detectAndCompute(img1, None)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Find Homography and Warp Perspective
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
    
        # Warp img1 to the size of img2 using the homography matrix
        h2, w2 = img2.shape
        warped_img1 = cv2.warpPerspective(img1, M, (w2, h2))
    
        # Combine the images by drawing the warped image onto the second image
        combined_img = cv2.addWeighted(img2, 0.3, warped_img1, 0.7, 0)
    
        # Display the combined image using OpenCV's imshow function
        cv2.imshow('Warped Image Combined', combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 10))
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display the matches using OpenCV's imshow function
    # cv2.imshow('Matches', img_matches)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
