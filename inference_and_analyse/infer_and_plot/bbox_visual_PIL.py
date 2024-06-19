from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from infer_and_plot.infer_and_plot import normalize_bbox

label_dict ={
        "biowaste" : 0,
        "cardboard" : 1,
        "electronic": 2,
        "glass" :3,
        "hazardous":4,
        "metal":5,
        "other":6,
        "paper":7,
        "plastic":8,
        "textile":9,
}

category_names = {v: k for k, v in label_dict.items()}

annotation_colors = {
    0: (164, 211, 156),     
    1: (176, 126, 68),    
    2: (72, 175, 142),     
    3: (113, 141, 77),   
    4: (239, 98, 98),   
    5: (232, 189, 4),   
    6: (120, 120, 120),     
    7: (176, 194, 209),     
    8: (255, 216, 206),     
    9: (89, 167, 191)  
}

def get_bbox_cat_from_res(result):
    
    predictions = result['predictions']['predictions']
    bboxes = []
    categories = []
    confidences = []
    
    for prediction in predictions:
        box = prediction['box']
        confidence = prediction['confidence']
        label = prediction['class']
        
        bboxes.append(box)
        confidences.append(confidence)
        categories.append(label)
    
    
    return bboxes,categories

def is_bbox_nomalized(bbox):
    if bbox[0] > 1 or bbox[1] > 1 or bbox[2] > 1 or bbox[3] > 1:
        return False
    return True
def resize_text(text_image,image_size):
    text_image_size = text_image.size
    text_image_ratio = text_image_size[0]/text_image_size[1]
    height = image_size[1]

    height = int(height/23)
    
    width = int(height * text_image_ratio)

    text_image = text_image.resize((width,height))

    return text_image

def draw_bboxes(image, bboxes, categories, category_names, annotation_colors):
    """
    Draw bounding boxes on the image with category names.

    Parameters:
        image (PIL.Image.Image): Input image.
        bboxes (list): List of bounding boxes, each in the format [x_min, y_min, x_max, y_max] normalized.
        categories (list): List of category indices corresponding to the bounding boxes.
        category_names (dict): Dictionary mapping category indices to category names.
        annotation_colors (dict): Dictionary mapping annotation IDs to colors.

    Returns:
        PIL.Image.Image: Image with bounding boxes and category names drawn.
    """
    draw = ImageDraw.Draw(image)
    w, h = image.size

    if not is_bbox_nomalized(bboxes[0]):

        bboxes = [normalize_bbox(bbox, (640, 640)) for bbox in bboxes]

        # we assume that the bbox is adapted to (640,640) and we normalize it to be between 0 and 1
    
    # Load the image to display

    # the bbox is in the format when yolo predicts things (x - center of the bbox, y - center of the bbox, width, height)
    # we need to convert it to the format (x_min, y_min, width, height) to draw it on the image

    for bbox, category in zip(bboxes, categories):
        x, y, width, height = bbox
        x_min, y_min = int((x-width/2) * w), int((y-height/2) * h)
        x_max, y_max = int((width/2+x) * w), int((height/2+y) * h)

        color = annotation_colors[category]
        label = str(category_names[category])
        # Load the image to display
        width = 5
        if h> 1500:
            width = 8
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=width)

        image_to_display_path = f'infer_and_plot/templates/{label}.png'
        
        
        if image_to_display_path.split('/')[-1] in os.listdir('infer_and_plot/templates'):
            image_to_display = Image.open(image_to_display_path).convert("RGBA")
        
        # Resize the image to display to fit within the bounding box
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            image_to_display_resized = resize_text(image_to_display,(w, h))
        
        # Calculate position to paste the image to display in the center of the bounding box
            paste_x = x_min + (bbox_width - image_to_display_resized.width) // 2
            paste_y = y_min + (bbox_height - image_to_display_resized.height) // 2

        # Paste the resized image into the bounding box
            image.paste(image_to_display_resized, (paste_x, paste_y), image_to_display_resized)

    return image

def visualize_image_with_annotations(image_path, result, annotation_colors=annotation_colors,save_to_file=False,save_path='prediction_'):
    """
    Visualize image with bounding boxes and category names.

    Parameters:
        image_path (str): Path to the input image.
        annotations_json (str): JSON string containing bounding boxes and categories.
        annotation_colors (dict): Dictionary mapping annotation IDs to colors.
        save_to_file (bool): Whether to save the image with annotations to a file, or display it in a window.(default: False)

    Returns:
        None
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Load annotations
    bboxes,categories = get_bbox_cat_from_res(result)
    
    # Define category names (adjust this according to your dataset)

    # Draw bounding boxes and category names
    image_with_annotations = draw_bboxes(image, bboxes, categories, category_names, annotation_colors)

    if save_to_file:
        if not os.path.exists('predictions'):
            os.makedirs('predictions')
        if save_path == 'prediction_':
            save_path = save_path + image_path.split('/')[-1]
        else:
            save_path = save_path.split('.')[0] + '.jpg'
        image_with_annotations.save(save_path)

    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(image_with_annotations)
        plt.axis('off')
        plt.show()

'''
image_path = '/Users/macbook/Desktop/Trashback/1689671638490-image1689671576159.jpg'  # Replace with the path to your image file

visualize_image_with_annotations(image_path, result)
'''

#image_path =  
result = {'predictions': {'predictions': [{'box': (0.559375,
     0.54609375,
     0.38125,
     0.2046875),
    'confidence': 0.5371050834655762,
    'class': 7}]}}

#visualize_image_with_annotations(image_path, result)


