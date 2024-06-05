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
from tqdm import tqdm
import json


def apply_mask_and_create_image(mask, original_image):
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
def calculate_black_transparent_percentage(image_path):
    # Load the image
    image = Image.open(image_path)
     
    # Convert the image to an RGBA image (if not already in that format)
    rgba_image = image.convert("RGBA")
    
    # Convert the RGBA image to a NumPy array
    image_array = np.array(rgba_image)
    
    # Calculate total pixels
    total_pixels = image_array.shape[0] * image_array.shape[1]
    
    # Calculate black pixels (where R, G, and B channels are all 0)
    black_pixels = np.sum((image_array[:,:,0] == 0) & (image_array[:,:,1] == 0) & (image_array[:,:,2] == 0))
    
    # Calculate transparent pixels (where alpha channel is 0)
    transparent_pixels = np.sum(image_array[:,:,3] == 0)
    
    # Calculate percentages
    black_percentage = (black_pixels / total_pixels) * 100
    transparent_percentage = (transparent_pixels / total_pixels) * 100
    
    return black_percentage, transparent_percentage

HOME = os.getcwd() # keep home as sys.argv but not use it for weights
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

path_unsegdir = os.path.join(HOME,"p2-unsegmentedBins")
path_config = os.path.join(HOME,"config.json")
with open(path_config, 'r') as file:
        config_data = json.load(file)
        # Retrieve the value of 'i0'
        i0 = config_data["i0"]
        
new_annotations_path = os.path.join(HOME,"p3-segments","p3anno.json")
if os.path.exists(new_annotations_path):
    with open(new_annotations_path, 'r') as file:
        new_annotations = json.load(file)
else:
    new_annotations = {}
    
    
for file in tqdm(os.listdir(path_unsegdir)):
    if file.endswith(('.png', '.jpg', '.jpeg')):
        original_file_path = os.path.join(path_unsegdir, file)
        extension = file.split('.')[-1]
        new_file_name = f'bin{i0}.{extension}'
        new_file_path = os.path.join(path_unsegdir, new_file_name)
        os.rename(original_file_path, new_file_path)
        IMAGE_PATH = new_file_path
        image_bgr = cv2.imread(IMAGE_PATH)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_result = mask_generator.generate(image_rgb)
        #mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        #detections = sv.Detections.from_sam(sam_result=sam_result)
        #annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        masks_bbox = [
            mask['segmentation'],mask['bbox']
            for mask
            in sorted(sam_result, key=lambda x: x['area'], reverse=True)
        ]
        masks = [
            mask['segmentation']
            for mask
            in sorted(sam_result, key=lambda x: x['area'], reverse=True)
        ]
        i=0
        #for image_mask,image_bbox in tqdm(masks_bbox, desc=f'Processing masks for {new_file_name}'):
        for image_mask in tqdm(masks, desc=f'Processing masks for {new_file_name}'):
            i+=1
            segment_array = apply_mask_and_create_image(image_mask, image_rgb)
            
            segment = cv2.cvtColor(segment_array, cv2.COLOR_RGB2BGR)
            mask_uint8 = (image_mask.astype(np.uint8) * 255)  # Convert boolean to uint8
            segment_mask = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
            
            save_path = os.path.join(HOME,"p3-segments","images", f'bin-bin{i0}-{i}-0.jpg') 
            save_path_b = os.path.join(HOME,"p3-segments","bmasks", f'bin-bin{i0}-{i}-0.jpg')
            cv2.imwrite(save_path, segment)
            cv2.imwrite(save_path_b, segment_mask)
            #update annotation
            new_annotations[f'bin-bin{i0}-{i}-0.jpg'] = {"bin" : f'bin{i0}',"label" : "unknown"} # ADD BBOX IN ANNOTATIONS ???

        shutil.move(os.path.join(HOME,"p2-unsegmentedBins",f'bin{i0}.{extension}'),
                    os.path.join(HOME,"p6-segmentedBins",f'bin{i0}.{extension}') )
        i0+=1
        
        
with open(new_annotations_path, 'w') as file: 
    json.dump(new_annotations, file, indent=4)
config_data['i0'] = i0
with open(path_config, 'w') as file:
    json.dump(config_data, file, indent=4)
    

        
"""
"img(.png?) = { "bin" : name_bin,
                "label":(label or "unknown" or "bg" or "other")
                "bbox" : [X,Y,W,H] #new here
}
"""
# A GERER COMMENT ON ECRIT LES ANNOTATIONS !!!!!
    
#Comment limiter malgré tout le nombre de masks sortis
#RGB BGR il y'a quelque chose qui se permute dans l'histoire à revoir

#Nouvelle nomenclature (bin-single)-(name_bin_label)-id-auid
#exemple bin-bin1-1265-0   single_glass_3432-2
# dans le gui validation enlever les images de bin qui ont déja été traitée ?


#cahier de charges :

#on a besoin d'avoir une config et de la modifier au fil de la génération de noms pour les poubelles
#uniformiser les formats, ce qu'on met dans les annotations ?jpg ou pas dans l'annotation.

#ou se trouvera ce script ? /preprocessing/binSegmentation =HOME
#ou se trouveront les weights ? os.path.join(HOME,"weights")
#unsegmented bin path = os.path.join(HOME,"p2-unsegmentedBins")
#segmented bin path = os.path.join(HOME,"p6-segmentedBins")
#segments path = os.path.join(HOME,"p3-segments")
#segments images path = os.path.join(HOME,"p3-segments","images")
#segments bmasks path = os.path.join(HOME,"p3-segments","bmasks")
#segments p3anno path = os.path.join(HOME,"p3-segments","p3anno.json")



#autre aspects 
#un gui pour classifier les labels pour préparer les données de RES et contourner les images de piètre qualité de rembg au cas ou ? 
#étudier le pourcentage optimal de données synthétique 
#faire une fonction python qui génère un env
#faire un nouveau domain pour cela ?
    

    


