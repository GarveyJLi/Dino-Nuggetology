from img_methods import *


def process_image(img_path):
    """
    Takes in the path of a JPG and creates a new image containing its edges.
    
    Parameters:
    img_path (str): Image path.
    
    Returns:
    img_edge (List): Edge image as a numpy array.
    
    """
    
    # Transforms to do: Mirror, rotate (15 degree increments)
    
    gray_img = load_gray_image(img_path)
    gauss_img = gaussian_lpf(gray_img, 42 ** 2)
    
    gmag, ang, gmag_nms = edge_detection(gauss_img)
    binary_gmag = gmag > 0.65
    original_edge = np.clip(binary_gmag * 255, 0, 255)
    
    return 

