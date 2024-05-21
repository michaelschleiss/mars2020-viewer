import numpy as np
import rasterio
from PIL import Image

def _read_image(path_to_image: str) -> np.ndarray[int, np.dtype[np.uint8]]:
  '''Reads image from geotiff and converts to numpy array. For now RGB images are expected.
  Output is in BGR. None Values are replaced with zero. Alpha channel can mask values that were None.'''
  with rasterio.open(path_to_image) as src:
    img = src.read()
    img[img == src.nodatavals[0]] = 0 # None values are converted to black pixels
  return img
  
filename = "/Users/michael/Downloads/JEZ_hirise_soc_007_orthoMosaic_25cm_Ortho_blend120.tif"
target_ground_resolution = 10
source_ground_resolution = 0.25

# read image
img = _read_image(filename)[0]

# crop image to remove black borders 
x0, x1 = int(85936 * 0.09), int(85936 * 0.87)
y0, y1 = int(81064 * 0.32), int(81064 * 0.82)
img  = img[y0:y1, x0:x1]

# scale image to desired target resolution
scale_factor = target_ground_resolution / source_ground_resolution
target_height, target_width = int(img.shape[0]/ scale_factor), int(img.shape[1] / scale_factor)
img = Image.fromarray(img).resize((target_width, target_height), Image.LANCZOS)

# show image and print infos
img.show()
print(img.size)

# save image
img.save(f"res/JEZ_hirise_{target_ground_resolution}m.png")
