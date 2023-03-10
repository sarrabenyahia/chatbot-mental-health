import pandas as pd
import json
import csv

# Read the CSV file
df = pd.read_csv("Dataset/Mental_Health_FAQ.csv")

# Create a list to store the JSON objects
intents = []

# Iterate over each row of the dataframe
for i, row in df.iterrows():
    # Create a dictionary for the current row
    intent = {}
    intent["tag"] = row["tag"]
    intent["patterns"] = row["patterns"].split("|")
    intent["responses"] = [sentence.replace("\n", " ") for sentence in row["responses"].split(". ")]
    intent["context"] = [""]
    # Check if the tag already exists in the list
    existing_intent = next((i for i in intents if i["tag"] == intent["tag"]), None)
    if existing_intent:
        # If the tag already exists, append the new patterns to the existing intent
        existing_intent["patterns"].extend(intent["patterns"])
    else:
        # If the tag does not exist, append the intent to the list
        intents.append(intent)

# Convert the list of intents to a JSON object
intents_json = json.dumps({"intents": intents})

# Write the JSON object to a file
with open("Dataset/intents.json", "w") as f:
    f.write(intents_json)

