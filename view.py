import pdr
import numpy as np
import glob
import cv2
from datetime import datetime
import re

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

# Read the .img file
for i in range(len(filenames)):
    data = pdr.read(filenames[i])
    cv2.imshow('VIEW', data['IMAGE'])
    cv2.waitKey(200)
    metadata = parse_lbl_content(data['IMAGE_HEADER'])
    print(f"======== Image {i} =================".zfill(2))
    # print(eetadata['ORIGIN_OFFSET_VECTOR'])
    # print(metadata['ORIGIN_ROTATION_QUATERNION'])
    # print(metadata['START_TIME'])
    time = datetime.strptime(metadata['START_TIME'], datetime_format)
    position = np.array(metadata['ORIGIN_OFFSET_VECTOR'].strip('()').split(','), dtype=float) 
    print(position - initial_position)
    print(time - initial_time)

