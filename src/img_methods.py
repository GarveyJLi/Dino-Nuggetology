import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import data
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

def load_gray_image(img_path) :
    img = Image.open(img_path).convert('L').resize((200, 200), resample=Image.LANCZOS)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data
  


def save_image(npdata, out_path) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L")
    img.save(out_path)
    
# Function for constructing gaussian lowpass filter in frequency domain
def gaussian_lpf_generator(var, P, Q):
  e_val = np.e
  
  freq_kernel = np.zeros((P, Q))
  for v in range(freq_kernel.shape[0]):
    for u in range(freq_kernel.shape[1]):
      # Distance from center
      d = np.sqrt(((u - P/2)**2) + ((v-Q/2)**2))
      
      freq_kernel[v, u] = e_val ** (-(d**2)/(2 * var))
  return freq_kernel

# function that applies gaussian low-pass filter in the frequency domain
def gaussian_lpf(img, var):
  """
  img: input image (H,W)
  var: variance for the gaussian lowpass filter 

  returns:
  out: output image after low-pass filter has been applied in the frequency domain
  """
  # your code here
  fft_img = np.fft.fftshift(np.fft.fft2(img))
  lpf = gaussian_lpf_generator(var, img.shape[0], img.shape[1])
  out = np.fft.ifft2(np.fft.ifftshift(np.multiply(fft_img, lpf)))

  return out.real

# function that performs edge detection
def edge_detection(img):
  """
  img: input image (H,W)

  returns: 
  gmag: gradient magnitude image (H,W)
  ang: angle image (H,W)
  gmag_nms: gradient magnitude image after non-maximal suppression (H,W)
  """
  # your code here
  H, W = img.shape  
  nimg = img / 255
  gsimg = gaussian_filter(nimg, sigma=0.5)
  sobel_y = [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
  sobel_x = [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
  gy = convolve2d(gsimg, sobel_y, mode='same', fillvalue=1)
  gx = convolve2d(gsimg, sobel_x, mode='same', fillvalue=1)
  gmag = np.sqrt(gy ** 2 + gx ** 2)
  ang = np.arctan2(gy, gx)
  
  gmag_nms = np.zeros([H, W])

  for y in range(H):
    for x in range(W):
      temp_mag = gmag[y, x]
      # Initialize neighbors for comparison
      p1 = float('-inf')
      p2 = float('-inf')
      a = ang[y, x]
      # Get neighbors for each respective direction bin
      if abs(a) <= 22.5 or abs(a) >= 157.5:
        try: p1 = gmag[y-1, x]
        except: pass
        try: p2 = gmag[y+1, x]
        except: pass
      elif (a > 22.5 and a < 67.5) or (a < -112.5 and a > -157.5):
        try: p1 = gmag[y-1, x-1]
        except: pass
        try: p2 = gmag[y+1, x+1]
        except: pass
      elif (abs(a) > 67.5 and abs(a) < 112.5):
        try: p1 = gmag[y, x-1]
        except: pass
        try: p2 = gmag[y, x+1]
        except: pass
      elif (a < -22.5 and a > -67.5) or (a > 112.5 and a < 157.5):
        try: p1 = gmag[y-1, x+1]
        except: pass
        try: p2 = gmag[y+1, x-1]
        except: pass
      # If magnitude is less than directional neighbors, suppress to 0
      if temp_mag < p1 or temp_mag < p2:
        temp_mag = 0
      gmag_nms[y, x] = temp_mag

  return gmag, ang, gmag_nms


# DO NOT USE THIS TO ROTATE AN IMAGE. USE rotate_image() instead.
def rotate_linear(img, h):
    """
    img: source image
    h: 2D transformation matrix

    returns:
    rot_img: image after rotation
    """
    # your code here
    img_hgt = img.shape[0]
    img_wdt = img.shape[1]

    rot_img = np.zeros(img.shape)

    for y in range(img_hgt):
        for x in range(img_wdt):
            # Finding the inverse coordinates of the destination coordinates
            # x = H-1x'
            inverse_coords = np.dot(h, np.array([x, y, 1]))
            xi, yi, _ = inverse_coords

            # Calculating x0, x1, y0, and y1 for linear interpolation
            x0 = np.int64(np.floor(xi))
            x1 = x0 + 1
            y0 = np.int64(np.floor(yi))
            y1 = y0 + 1
          
            # Checking if x0, x1, y0, y1 are in the original image
            if (x0 >= 0 and x0 < img_wdt and y0 >= 0 and y0 < img_hgt and 
                x1 >= 0 and x1 < img_wdt and y1 >= 0 and y1 < img_hgt):
                # Calculating pixel values between x0 and x1 for each at xi
                # for each y0, y1
                a = (x1-xi) * img[y0][x0] + (xi-x0) * img[y0][x1]
                b = (x1-xi) * img[y1][x0] + (xi-x0) * img[y1][x1]
                # Calculating pixel value between a and b at yi
                pixel_val = (y1-yi) * a + (yi-y0) * b
                # Setting pixel value at destination coordinates to 
                # previously calculated pixel value
                rot_img[y][x] = np.round(pixel_val)

    return rot_img
  
  
  # function that calculates the 2D transformation matrix for rotating an image about its center
def get_transformation_matrix(img_hgt, img_wdt, rot):
     """
     input:
     img_ht: image height in pixels
     img_wt: image width in pixels
     rot: rotation angle in radians

     output: 
     h: 2D transformation matrix
     """
     # your code here
     center_y, center_x = np.array([img_hgt, img_wdt]) // 2

     # Translation transformation matrix to map center to origin
     ht = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])

     # Rotation transformation matrix about the origin
     hr = np.array([[np.cos(rot), -np.sin(rot), 0], 
          [np.sin(rot), np.cos(rot), 0],
          [0, 0, 1]])
     
     # Inverse translation of the translation transformation matrix above
     ht_inv = np.linalg.inv(ht)

     # H = Ht * Hr * Ht-1
     h = ht@hr@ht_inv
  
     return h    
   
   
def rotate_image(img, deg):
    return rotate_linear(img, get_transformation_matrix(img.shape[0], img.shape[1], np.deg2rad(deg))) 
   
   
   