from resnet import predict_classification_utils
from sam import sam_utils
import os

HOME = os.getcwd()

res2 = sam_utils.sam_folder(HOME)
res = predict_classification_utils.predict_folder(os.path.join(HOME,"p3-segments","images")) # modifier cette fonction pour qu'elle fasse les changements nécéssaires dans la pipeline de dossiers