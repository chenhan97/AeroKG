from transformers import AutoTokenizer, AutoModelForTokenClassification
import requests


class NERModel():
    def __init__(self):
        self.token = "Bearer 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"
        self.API_URL = "https://api-inference.huggingface.co/models/bluenqm/AerospaceNER"
        self.headers = {"Authorization": self.token}

    def inference(self, inputs):
        response = requests.post(self.API_URL, headers=self.headers, json=inputs)
        return response.json()

'''
a = model.inference("In addition to manufacturing major components for Typhoon, the site builds the aft fuselage and the horizontal and vertical tail planes for every F-35 military aircraft under contract to the prime contractor, Lockheed Martin.")
print(a)'''