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
from .utils import load_sam_model,occ_mem_info,filter_nested_similar_masks,apply_mask_and_create_image
from .utils import save_segmentation_results,update_annotations,open_config_annotations,save_config_annotations,get_segment_mask,process_image_with_filter,rename_unsegmented_bin



def segment_objects_in_folder(HOME):
    """
    ARGS :
        HOME absolute path of pipeline/preprocessing/binSegmentation
    Returns:
        None
        
    will take bin pictures in p2 segment them and put segments in p3 and move segmented bins to p6 while modifying annotations and config
    
    """
    
    #LOADING SAM
    torch.cuda.empty_cache()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam,mask_generator =load_sam_model("vit_l",os.path.join("sam","weights"), DEVICE )

    #DEFINING LOCAL PATHS
    path_unsegmentedBins = os.path.join(HOME,"p2-unsegmentedBins")
    path_config = os.path.join(HOME,"config.json")
    new_annotations_path = os.path.join(HOME,"p3-segments","p3anno.json")
    output_dir = os.path.join(HOME,"p3-segments") 
    segmented_bins_dir = os.path.join(HOME,"p6-segmentedBins")
    
    #LOADING CONFIG AND ANNOTATIONS
    config_data,bin_id,new_annotations = open_config_annotations(path_config,new_annotations_path)
        
    #PROCESSING BINS IMAGES WITH SAM
    for file in tqdm(os.listdir(path_unsegmentedBins)):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            config_data,bin_id,new_annotations = open_config_annotations(path_config,new_annotations_path)
            image_path = rename_unsegmented_bin(bin_id,file,path_unsegmentedBins)
            masks_bbox,image_rgb = process_image_with_filter(image_path, mask_generator)
            
            new_annotations = save_segmentation_results(masks_bbox,
                                                               image_rgb,
                                                               bin_id,
                                                               output_dir,
                                                               path_unsegmentedBins,
                                                               segmented_bins_dir,
                                                               new_annotations,
                                                               new_annotations_path)
            bin_id+=1
            config_data['bin_id'] = bin_id
            save_config_annotations(path_config,
                                    new_annotations_path,
                                    config_data,
                                    new_annotations)
            del image_rgb,masks_bbox,new_annotations
            torch.cuda.empty_cache()
        



    del sam,mask_generator
    torch.cuda.empty_cache()
    
    


if __name__ == "__main__":
    # Check if the command line argument is provided
    if len(sys.argv) > 1:
        path = sys.argv[1]
        sam_folder(path)
    else:
        print("Usage: sam_folder.py <path to file>")

    

    


