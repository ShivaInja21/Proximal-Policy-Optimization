import numpy as np
import cv2

def display_pendulum_image(angle, image_size=64):
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    center = (image_size // 2, image_size // 2)  
    length = image_size // 2 - 4  
    thickness = 3  

    end_x = int(center[0] + length * np.sin(angle))
    end_y = int(center[1] - length * np.cos(angle))  

    cv2.line(image, center, (end_x, end_y), color=255, thickness=thickness)

    return image / 255.0  
