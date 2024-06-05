import os
import torch
import torchvision
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import numpy as np
from PIL import Image
import shutil
from tqdm.auto import tqdm
import json
from typing import Any
from datetime import datetime

def apply_mask_and_create_image(mask, original_image):
    """
    mask : binary mask of the object
    original_image : image with bg + the differents objects
    return the segmented image in np.array format.
    """
    # Convert the mask to a binary array (assuming it's already binary, otherwise adjust)
    binary_mask = np.array(mask, dtype=bool)
    
    # Create a new array filled with zeros (black) with the same shape as the original image
    new_image_array = np.zeros_like(original_image)
    
    # Apply the mask to each channel of the image
    for c in range(3):  # Assuming 3 channels: R, G, B
        new_image_array[:,:,c] = original_image[:,:,c] * binary_mask
    #new_image = Image.fromarray(new_image_array.astype('uint8'))
    return new_image_array
    #return new_image
def calculate_black_percentage(image_array):
    """
    DEPRECATED !
    take an image as a np.array as input
    return percentage of black pixels , percentage of transparency
    """
    total_pixels = image_array.size  # Total number of pixels in the mask
    black_pixels = np.sum(image_array == 0)  # Sum of pixels that are 0 (background)
    
    # Calculate percentages
    black_percentage = (black_pixels / total_pixels) * 100
    return black_percentage
def _calculate_intersection_score(elem1,elem2):
    #elem1: dict[str, Any], elem2: dict[str, Any]
#) -> float:
    """Calculates the intersection score for two masks.

      Args:
        elem1: The first element.
        elem2: The second element.

      Returns:
        The intersection score calculated as the ratio of the intersection
        area to the area of the smaller mask.
      """

      # Check if the masks have the same dimensions.
    if elem1['segmentation'].shape != elem2['segmentation'].shape:
        raise ValueError('The masks must have the same dimensions.')

    min_elem = elem1 if elem1['area'] < elem2['area'] else elem2
    intersection = np.logical_and(elem1['segmentation'], elem2['segmentation'])
    score = np.sum(intersection) / np.sum(min_elem['segmentation'])
    return score
def is_bbox_ok(bbox,img_size,margin = 5):
    
    '''
    input: 
    - bounding box of the mask 
    - size of the image
    - margin as a parameter
    
    output:
    True if the bounding box in in the inside of the image
    False if it touches one of the higher and lower edges, of both lateral edges
    
    A margin of 10 for images of 
    
    '''
    xmin,ymin,bb_w,bb_h = bbox
    
    xmax = xmin + bb_w
    ymax = ymin + bb_h

    w,h = img_size
    
    touching_sides = 0
    # Check each side to see if the bounding box touches it with the given margin
    if xmin <= margin:
        touching_sides += 1
    if xmax >= w - margin:
        touching_sides += 1
    if ymin <= margin:
        touching_sides += 1
        return False
    if ymax >= h - margin:
        return False
        touching_sides += 1
    
    if touching_sides < 2:
        return True
    
    return False
def filter_nested_similar_masks(elements): #DEPCRECATED
#    elements: list[dict[str, Any]]
#) -> list[dict[str, Any]]:
    
    """
    DEPRECATED
    Filters out nested masks from a list of elements.
    This function does not filter masks based on BBOX criterion (touching side)

      Args:
        elements: A list of dictionaries representing elements.

      Returns:
        A list of dictionaries representing elements with nested masks filtered out.
      """
    restemp =[]
    retained_elements = []
    handled_indices = (set())  # To keep track of indices that have already been handled
    for i, elem in enumerate(elements):
        bp=calculate_black_percentage(np.array(elem['segmentation'], dtype=bool))
        if 99.7<bp:
            continue
        if bp < 50:
            continue
        restemp.append(elem)

    for i, elem in enumerate(restemp):
        if i in handled_indices:
            continue  # Skip elements that have already been handled

        matching_indices = [i]  # Start with the current element

        # Find all elements that match with the current element
        for j, other_elem in enumerate(restemp):
            if i != j and _calculate_intersection_score(elem, other_elem) > 0.95:
                matching_indices.append(j)

        # If more than one element matched, find the one with the highest 'area'
        # and add it to retained_elements
        if len(matching_indices) > 1:
            highest_area_index = max(matching_indices, key=lambda idx: restemp[idx]['area'])
            retained_elements.append(elements[highest_area_index])
            handled_indices.update(matching_indices)  # Mark all matching indices as handled
        else:
            # If no matches were found, retain the current element
            retained_elements.append(elem)
            handled_indices.add(i)  # Mark the current index as handled

    return retained_elements
def filter_nested_similar_and_small_masks(elements,total_area):# to use
#    elements: list[dict[str, Any]]
#) -> list[dict[str, Any]]:
    
    """Filters out nested masks from a list of elements.

      Args:
        elements: A list of dictionaries representing elements.

      Returns:
        A list of dictionaries representing elements with nested masks filtered out.
      """
    restemp =[]
    retained_elements = []
    handled_indices = (set())  # To keep track of indices that have already been handled
    filtered = []
    
    img_size = elements[0]['crop_box'][2:4]
    for i, elem in enumerate(elements):
        bbox = elem['bbox']
        
        if not is_bbox_ok(bbox,img_size,margin=10):
            filtered.append(elem)
            continue
        
        #bp=calculate_black_percentage(np.array(elem['segmentation'], dtype=bool))
        segment_area = elem['area']
        bp = 100 - (segment_area / total_area) * 100
        if 99.7<bp:
            continue
        if bp < 50:
            continue
        restemp.append(elem)
        
    restemp = sorted(restemp, key=lambda x: x['area'], reverse=True)
    
    for i, elem in enumerate(restemp):
        if i in handled_indices:
            continue  # Skip elements that have already been handled
        
        #if there is a match with a filtered element, we do not take the object
        for j, filtered_elem in enumerate(filtered):
            if i != j and _calculate_intersection_score(elem, filtered_elem) > 0.95:
                handled_indices.add(i)
            
        if i in handled_indices:
            continue
            
        matching_indices = [i]  # Start with the current element
        
        # Find all elements that match with the current element
        for j, other_elem in enumerate(restemp):
            if i != j and _calculate_intersection_score(elem, other_elem) > 0.95:
                matching_indices.append(j)

        # If more than one element matched, find the one with the highest 'area'
        # and add it to retained_elements
        if len(matching_indices) > 1:
            highest_area_index = max(matching_indices, key=lambda idx: restemp[idx]['area'])
            retained_elements.append(restemp[highest_area_index])
            handled_indices.update(matching_indices)  # Mark all matching indices as handled
        else:
            # If no matches were found, retain the current element
            retained_elements.append(elem)
            handled_indices.add(i)  # Mark the current index as handled
    del filtered,restemp
    torch.cuda.empty_cache()

    return retained_elements

def occ_mem_info():
    """
    return the percentage of occupied gpu memory space
    for debugging purpopses
    """
    x,y = torch.cuda.mem_get_info()
    return str((1-x/y)*100)
def load_sam_model(model_type,checkpoint_home, DEVICE ):
    """Charge le modèle SAM spécifié et le générateur de masques automatiques.
        checkpoint_home (str): Le chemin d'accès au fichier de point de contrôle du modèle.
        device (torch.device): L'appareil sur lequel exécuter le modèle (CPU ou GPU).

    Args:
        model_type (str): Le type de modèle SAM à charger ('vit_h', 'vit_l', 'vit_b').    
    Returns:
        sam (Sam): Le modèle SAM chargé.
        mask_generator (SamAutomaticMaskGenerator): Le générateur de masques automatiques.
    """
    
    
    if model_type == "vit_l":
        CHECKPOINT_PATH = os.path.join(checkpoint_home, "sam_vit_l_0b3195.pth")
    if model_type == "vit_h":
        CHECKPOINT_PATH = os.path.join(checkpoint_home, "sam_vit_h_4b8939.pth")
    if model_type == "vit_b":
        CHECKPOINT_PATH = os.path.join(checkpoint_home, "sam_vit_b_01ec64.pth")
    
    with torch.no_grad():
        sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
        mask_generator = SamAutomaticMaskGenerator(sam)
    return sam,mask_generator
def process_image_with_filter(image_path, mask_generator):
    """Traite une image en utilisant le générateur de masques automatiques et renvoie les masques de segmentation.

    Args:
        image_path : le path de l'image à traiter.
        mask_generator (SamAutomaticMaskGenerator): Le générateur de masques automatiques.


    Returns:
        masks_bbox (List[Tuple[np.ndarray, List[int]]]): Une liste de tuples contenant les masques de segmentation et leurs boîtes englobantes.
        image_rgb : l'image de la corbeille initiale
    """
    image_bgr = cv2.imread(image_path)
    cv2.resize(image_bgr, (620, 620))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    total_area = height * width
            
    #SAM Inference :
    with torch.no_grad():
        sam_result = mask_generator.generate(image_rgb)
    if sam_result:    
        fsam_result = filter_nested_similar_and_small_masks(sam_result,total_area)
               
    masks_bbox = [
        (mask['segmentation'],mask['bbox'])
        for mask
        in sorted(fsam_result, key=lambda x: x['area'], reverse=True)
    ]
    del sam_result,fsam_result
    torch.cuda.empty_cache()
    return masks_bbox,image_rgb

def process_image_full_dict(image_path, mask_generator):
    """Traite une image en utilisant le générateur de masques automatiques et renvoie les masques de segmentation.

    Args:
        image_path : le path de l'image à traiter.
        mask_generator (SamAutomaticMaskGenerator): Le générateur de masques automatiques.


    Returns:
        sam_result: le dictionnaire résultant de l'inférance de sam 
    """
    image_bgr = cv2.imread(IMAGE_PATH)
    cv2.resize(image_bgr, (620, 620))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
    #SAM Inference :
    with torch.no_grad():
        sam_result = mask_generator.generate(image_rgb)
    return sam_result
def rename_unsegmented_bin(bin_id,file,path_unsegmentedBins):
    """
    Fonction qui renomme une image dans le dossier unsegmentedBins suivant les convention de nommage et avec le bin_id correspondant.
    Args:
        bin_id id of the bin processed
        file (str) nom du fichier image à traiter
        path_unsegmentedBins : le path du dossier des poubelles à segmenter (p2-unsegmentedBins)
    Returns:
        IMAGE_PATH : the new path of the current picture processed.
    
    """
    original_file_path = os.path.join(path_unsegmentedBins, file)
    extension = file.split('.')[-1]
    new_file_name = f'bin{bin_id}.{extension}'
    new_file_path = os.path.join(path_unsegmentedBins, new_file_name)
    os.rename(original_file_path, new_file_path)
    IMAGE_PATH = new_file_path
    return IMAGE_PATH
    
def get_segment_mask(image_mask,image_rgb):
    """
    transforme les np.array en images 
    Args :
        image_mask : les masques d'images c-a-d les elem['segmentations'] de sam
        image_rgb : l'image traitée de la corbeille.
    Returns:
        segment : the segmented image
        segment_mask : the binary mask of the segmented image
        
    """
    segment_array = apply_mask_and_create_image(image_mask, image_rgb)
    segment = cv2.cvtColor(segment_array, cv2.COLOR_RGB2BGR)
    mask_uint8 = (image_mask.astype(np.uint8) * 255)  # Convert boolean to uint8
    segment_mask = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    return segment,segment_mask
def save_config_annotations(path_config,new_annotations_path,config_data,new_annotations):
    with open(path_config, 'w') as file:
        json.dump(config_data, file, indent=4)
    with open(new_annotations_path, 'w') as file: 
        json.dump(new_annotations, file, indent=4)
def open_config_annotations(path_config,new_annotations_path):
    with open(path_config, 'r') as file:
        config_data = json.load(file)
        bin_id = config_data["bin_id"]
    if os.path.exists(new_annotations_path):
        with open(new_annotations_path, 'r') as file:
            new_annotations = json.load(file)
    else:
        new_annotations = {}
    return config_data,bin_id,new_annotations
def update_annotations(bin_id, i, image_bbox,new_annotations_path,new_annotations):
    """Met à jour les annotations JSON avec les informations de l'image segmentée.

    Args:
        bin_id (str): L'identifiant de la corbeille associée à l'image.
        i (int): L'index de l'image segmentée.
        new_annotations_path (str): Le chemin d'accès au fichier d'annotations JSON.
        new_annotations (dict) : le JSON d'annotations actuel
    Returns :
        new_annotations (dict) : le JSON d'annotations actualisé.
    
    """
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    new_annotations[f'bin-bin{bin_id}-{i}-0.jpg'] = {
        "bin" : f'bin{bin_id}.jpg',
        "bbox":image_bbox,
        "clabel" : "unknown",
        "mlabel":"unknown",
        "step":"segment",   #segment, truesegment, background, object (after manual labeling or classification)
        "timestamp": date_string
    }
    return new_annotations
    
    
def save_segmentation_results(masks_bbox, image_rgb, bin_id, output_dir,path_unsegmentedBins,segmented_bins_dir,new_annotations,new_annotations_path):
    
    """Enregistre les masques de segmentation et les images segmentées dans des fichiers.

    Args:
        masks_bbox (List[Tuple[np.ndarray, List[int]]]): Une liste de tuples contenant les masques de segmentation et leurs boîtes englobantes.
        image_rgb (np.ndarray): L'image d'origine en format RGB.
        bin_id (int): L'identifiant de la corbeille associée à l'image.
        path_unsegmentedBins : le path du dossier des poubelles à segmenter (p2-unsegmentedBins)
        output_dir (str): Le répertoire de sortie où enregistrer les masques et les images segmentées.
        segmented_bins_dir (str) : le répértoire où l'on stocke les images de corbeilles déjà segmentées.
        new_annotations (dict) : le JSON d'annotations actuel
        new_annotations_path (str): Le chemin d'accès au fichier d'annotations JSON.
        
    Returns :
       
        new_annotations (dict) : le JSON d'annotations actualisé
        
    """
    i=0
    for (image_mask,image_bbox) in tqdm(masks_bbox, desc=f'Processing masks for bin {bin_id}'):
        i+=1
        segment,segment_mask = get_segment_mask(image_mask,image_rgb)
        save_path = os.path.join(output_dir,"images", f'bin-bin{bin_id}-{i}-0.jpg') 
        save_path_b = os.path.join(output_dir,"bmasks", f'bin-bin{bin_id}-{i}-0.jpg')
        cv2.imwrite(save_path, segment)
        cv2.imwrite(save_path_b, segment_mask)
        
        #update annotation
        new_annotations = update_annotations(bin_id, i, image_bbox,new_annotations_path,new_annotations)
        

    shutil.move(os.path.join(path_unsegmentedBins,f'bin{bin_id}.jpg'),
                os.path.join(segmented_bins_dir,f'bin{bin_id}.jpg') )
    
    
    #del segment,segment_mask
    torch.cuda.empty_cache()
    return new_annotations
   
     
    