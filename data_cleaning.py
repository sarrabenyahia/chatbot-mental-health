import json
import re

def clean_data(data):

    # remove any duplicated patterns or responses
    for intent in data["intents"]:
        intent["patterns"] = list(set(intent["patterns"]))
        intent["responses"] = list(set(intent["responses"]))
        
    # remove the string "\u..."" from patterns and responses
    for intent in data["intents"]:  
        intent["patterns"] = [pattern.replace("\u00e2\u20ac\u2122", "") for pattern in intent["patterns"]]
        intent["responses"] = [response.replace("\u00e2\u20ac\u2122", "") for response in intent["responses"]]
        intent["responses"] = [response.replace("\u2013", "") for response in intent["responses"]]
        intent["responses"] = [response.replace("\u2014", "") for response in intent["responses"]]
        intent["responses"] = [response.replace("\u2018", "") for response in intent["responses"]]
        intent["responses"] = [response.replace("\u2019", "") for response in intent["responses"]]
        intent["responses"] = [response.replace("\u201c", "") for response in intent["responses"]]
        intent["responses"] = [response.replace("\u201d", "") for response in intent["responses"]]
 
    # check for empty patterns or responses and assign empty list
    for intent in data["intents"]:
        if not intent["patterns"]:
            intent["patterns"] = []
        if not intent["responses"]:
            intent["responses"] = []

    return data

with open("Dataset/updated_mentalhealth.json", "r") as f:
    data = json.load(f)

cleaned_data = clean_data(data)

with open("Dataset/cleaned_mentalhealth.json", "w") as f:
    json.dump(cleaned_data, f, indent=2)
