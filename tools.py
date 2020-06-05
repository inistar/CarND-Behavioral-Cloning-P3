import cv2

def read_image_rgb(image_path):    
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def convert_image_rgb(image):    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def remove_noise(image):
    return cv2.GaussianBlur(image, (3,3), 0)

def resize_image(image):
    image = image[50:140,:,:]
    image = cv2.resize(image,(200, 66))
    return image