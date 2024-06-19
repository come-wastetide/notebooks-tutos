import requests
from datetime import datetime
import json
from infer_and_plot import upload_image


class PredictionRecord:
    # We make a class for each prediction (! we work with image names and not id !)
    def __init__(self, image_name, prediction, model_name, user_id):
        self.image_name = image_name
        self.prediction = prediction
        self.model_name = model_name
        self.date = datetime.now()
        self.user_id = user_id

    def __repr__(self):
        return f"PredictionRecord(image_name={self.image_name}, prediction={self.prediction}, model_name={self.model_name}, date={self.date}, user_id={self.user_id})"

#the url of each model is stored in that dictionnary 
model_info = {
        "yolo-siom-v1": "http://18.204.8.241:8000/upload/"
    }
class ModelServer:
    #We also make a class for the model we're using, liked to a specific server, hence the name
    def __init__(self, model_name):
        self.model_name = model_name
        self.server_url = model_info[model_name]

    def send_image(self, image_path):
        response = upload_image(image_path,self.server_url)
        return response

    def get_prediction(self, image_path, user_id):
        prediction = self.send_image(image_path)
        image_name = image_path.split("/")[-1]
        record = PredictionRecord(image_name, prediction, self.model_name, user_id)
        return record


def prediction_list_from_records(records):
    #This function makes a list of predictions (dicts) from a list or records
    prediction_list = []
    for record in records:
        prediction_list.append(record.prediction)
    return prediction_list

def main(model_name, image_path, user_id):
    # Get the server URL
    # Create a ModelServer instance
    model_server = ModelServer(model_name)
    
    # Get prediction
    prediction_record = model_server.get_prediction(image_path, user_id)
    
    # Print the prediction record
    print(prediction_record)

if __name__ == "__main__":
    # Example usage
    model_info = {
        "yolo-siom-v1": "http://18.204.8.241:8000/upload/"
    }
    
    # Parameters
    model_name = "yolo-siom-v1"
    image_path = "/Users/macbook/Desktop/Trashback/1689690390941-user.jpg"
    user_id = "user123"
    
    # Run main function
    main(model_name, image_path, user_id)


