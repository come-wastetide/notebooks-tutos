import requests
import base64
import os
import json
import matplotlib.pyplot as plt


def upload_image(image_path, url, return_json=True):
    '''
    input :
    - path of the image
    - url of the server that returns a dict

    ouput :
    - a dict with predictions for the image
    '''

    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return None

    try:
        # Lire l'image et la convertir en binaire
        with open(image_path, "rb") as image_file:
            binary_image = image_file.read()

        # Convertir l'image binaire en une chaîne encodée en base64
        encoded_image = base64.b64encode(binary_image).decode('utf-8')

        # Créer le payload JSON
        payload = {
            "data": encoded_image
        }

        # Envoyer la requête POST
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)

        # Afficher la réponse
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        if return_json:
            return response.json()
        return response

    except Exception as e:
        print(f"Error: {e}")
        return None

def normalize_bbox(bbox,image_size):
    '''
    inputs :
    - a bbox not normalized (x,y,h,w) (typically 640x640 for yolo output)
    - the size of the original image (640x640 here or 1024 for PaliGemma as an example)

    output:
    - (x, y, w, h) between 0 and 1

    !! This does not tell us whether the bbox is in the format (x,y,w,h) (-> typical yolo output) or (x_min, y_min, w, h) (-> our convention) (or (x_min, y_min, x_max, y_max) !!
    '''
    x_min, y_min, w, h = bbox
    x = x_min / image_size[0]
    y = y_min / image_size[1]
    w = w/ image_size[0]
    h = h / image_size[1]
    return (x, y, w, h)

def normalize_prediction(prediction, image_size):
    '''
    inputs :
    - a prediction in the format of the output of the yolo model
    - the size of the original image (640x640 here or 1024 for PaliGemma as an example)

    output:
    - a prediction in the format of our convention (x_min, y_min, w, h) between 0 and 1
    '''
    for pred in prediction["predictions"]["predictions"]:
        if not is_bbox_nomalized(pred["box"]):
            pred["box"] = normalize_bbox(pred["box"], image_size)

    return prediction

# correspondances between labels and categories
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

# we store the categories that go together
keys = {"recyclable_keys": [1, 3, 5, 7, 8], "compostable_keys": [0], "hazardous_keys": [2, 4]}

category_names = {v: k for k, v in label_dict.items()}

#stadard colors for visualisation
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

def get_area(box):
    return box[2] * box[3]

# we then want to give a proportion of each class in the image 

def get_class_proportion(preds,give_index = False,normalise = True):
    class_proportion = {}
    for pred in preds["predictions"]["predictions"]:
        label = category_names[pred["class"]]
        if give_index:
            label = pred["class"]
        if label in class_proportion:
            class_proportion[label] += get_area(pred["box"])*pred["confidence"]
        else:
            class_proportion[label] = get_area(pred["box"])*pred["confidence"]

    #we then want to normalise the values
    total_area = sum(class_proportion.values())
    if normalise:
        for key in class_proportion:
            class_proportion[key] = class_proportion[key] / total_area

    # we round the results to 2 decimal places
        for key in class_proportion:
            class_proportion[key] = round(class_proportion[key], 2)
    return class_proportion  

# we create a function to plot a pie chart of the class proportion
# the colors come from the dictionary annotation_colors

def plot_pie_chart(class_proportion):
    fig, ax = plt.subplots()
    
    # Ensure colors are within the 0-1 range if they are RGBA values
    colors = []
    for label in class_proportion.keys():
        color = annotation_colors[label_dict[label]]
        if isinstance(color, tuple) and all(0 <= val <= 1 for val in color):
            colors.append(color)
        else:
            # Convert color to normalized RGBA if needed
            color = [val / 255.0 if val > 1 else val for val in color]
            colors.append(tuple(color))

    ax.pie(class_proportion.values(), labels=class_proportion.keys(), autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
 

def is_recyclable(composition,recyclable_keys = keys['recyclable_keys'],threshold=0.90):
    recyclable_rate = 0
    for i in composition.keys():
        if i in recyclable_keys:
            recyclable_rate += composition[i]
    return recyclable_rate >= threshold

def is_compostable(composition,compostable_keys= keys['compostable_keys'],threshold=0.90):
    compostable_rate = 0
    for i in composition.keys():
        if i in compostable_keys:
            compostable_rate += composition[i]
    return compostable_rate >= threshold

def is_hazardous(composition,hazardous_keys =keys['hazardous_keys'],threshold=0.05):
    hazardous_rate = 0
    for i in composition.keys():
        if i in hazardous_keys:
            hazardous_rate += composition[i]
    return hazardous_rate >= threshold

def get_waste_type(prediction):

    composition = get_class_proportion(prediction, give_index=True)
    if is_recyclable(composition):
        return "well sorted -> recyclable"
    elif is_compostable(composition):
        return "well sorted -> compostable"
    elif is_hazardous(composition):
        return "hazardous"
    else:
        return "not well sorted"

def get_global_composition(prediction_list):

    global_composition = {}
    for i in range(len(prediction_list)):
        composition = get_class_proportion(prediction_list[i], normalise=False)
        for key in composition.keys():
            if key in global_composition:
                global_composition[key] += composition[key]
            else:
                global_composition[key] = composition[key]
    #we then want to normalise the values
    total_area = sum(global_composition.values())
    for key in global_composition:
        global_composition[key] = global_composition[key] / total_area
    # we round the results to 2 decimal places
    for key in global_composition:
        global_composition[key] = round(global_composition[key], 2)
    return global_composition

#the prices here (€/kg) are based on price given be retailers. These are estimations and are subject to change 
price_dict = {
    'biowaste':0,
    'cardboard':0.083,
    'electronic':0,
    'glass':0.02391,
    'hazardous':0,
    'metal':0.2,
    'other':0,
    'paper':0.09552,
    'plastic':0.150,
    'textile':0
}

def calculate_price_per_kg(global_composition):
    price = 0
    for key in global_composition:
        price += global_composition[key] * price_dict[key]
    
    #we round the price to 2 decimal places
    price = round(price, 3)
    return price

CO2_dict = {
    'biowaste':0.385,
    'cardboard':0,
    'electronic':0,
    'glass':1.9,
    'hazardous':0,
    'metal':1.5,
    'other':0,
    'paper':0,
    'plastic':2.7,
    'textile':7.5
}

def calculate_CO2eq_per_kg(global_composition):
    CO2 = 0
    for key in global_composition:
        CO2 += global_composition[key] * CO2_dict[key]
    
    #we round the price to 2 decimal places
    CO2 = round(CO2, 3)
    return CO2


def summarize(prediction_list):
    waste_type_list = []
    global_composition = get_global_composition(prediction_list)
    for i in range(len(prediction_list)):
        waste_type_list.append(get_waste_type(prediction_list[i]))
    
    return waste_type_list, global_composition

def plot_summary(prediction_list):
    waste_type_list, global_composition = summarize(prediction_list)
    print(f'number of images: {len(prediction_list)}')
    plot_pie_chart(global_composition)

    # we print a pie chart for the waste type of each image
    fig, ax = plt.subplots()
    waste_type_dict = {}    
    for waste_type in waste_type_list:
        if waste_type in waste_type_dict:
            waste_type_dict[waste_type] += 1
        else:
            waste_type_dict[waste_type] = 1
    ax.pie(waste_type_dict.values(), labels=waste_type_dict.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
    print(f'The waste is worth {calculate_price_per_kg(global_composition)} euros per kg')
    yearly_worth = 254*calculate_price_per_kg(global_composition)
    yearly_worth = round(yearly_worth,1)

    print(f'This correspond to {yearly_worth} euros per year if you recycle 250 kg of waste per year')

    print(f'The waste represents {calculate_CO2eq_per_kg(global_composition)} kgCO2eq per kg of excess if not recycled')

    yearly_CO2 = 254*calculate_CO2eq_per_kg(global_composition)
    yearly_CO2 = round(yearly_CO2,1)
    print(f'This correspond to {yearly_CO2} kgCO2eq per year if you recycle 250 kg of waste per year')

def save_predictions_to_file(prediction_list, file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []

    data.extend([pred for pred in prediction_list])

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


