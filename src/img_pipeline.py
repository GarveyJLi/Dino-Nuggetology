from img_methods import *
from scipy.ndimage import zoom
import imutils

black_border = np.zeros((200, 200))
black_border[0:10, :] = 1
black_border[190:200, :] = 1
black_border[:, 0:10] = 1
black_border[:, 190:200] = 1

def process_image(img_path):
    """
    Takes in the path of a JPG and creates a new image containing its edges.
    
    Parameters:
    img_path (str): Image path.
    
    Returns:
    img_edge (List): Edge image as a numpy array.
    
    """
    
    # Transforms to do: Mirror, rotate (15 degree increments)
    
    rotation_degrees = np.arange(0, 360, 15)
    
    gray_img = load_gray_image(img_path)
    gauss_img = gaussian_lpf(gray_img, 42 ** 2)
    
    gmag, _, _ = edge_detection(gauss_img)
    binary_gmag = (gmag > 0.65) - black_border
    original_edge = np.clip(binary_gmag * 255, 0, 255)
    mirrored = np.flip(original_edge)
    
    all_imgs = []
    for deg in rotation_degrees:
        all_imgs.append(imutils.rotate(original_edge, angle=deg))
    for deg in rotation_degrees:
        all_imgs.append(imutils.rotate(mirrored, angle=deg))
    
    return all_imgs

