import json
import nltk
from nltk.corpus import wordnet
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):

    """
    This function uses a GPT-3 model to generate paraphrased sentences from a given input sentence. It takes the following arguments:

    model: the GPT-3 model to use for generating the paraphrased sentences
    tokenizer: the tokenizer used to tokenize the input sentence and decode the generated sentences
    sentence: the input sentence to be paraphrased
    num_return_sequences: the number of paraphrased sentences to generate
    num_beams: the number of beams to use in the beam search algorithm
    The function returns a list of paraphrased sentences in the form of strings.
    """

    # tokenize the text to be form of a list of token IDs
    inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
    # generate the paraphrased sentences
    outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
    )
    # decode the generated sentences using the tokenizer to get them back to text
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Load the json file
with open('Dataset/intents.json') as json_file:
    data = json.load(json_file)

# Loop through the intents
for intent in data['intents']:
    patterns = intent['patterns']
    new_patterns = []
    # Loop through the patterns
    for pattern in patterns:
        synonyms = get_paraphrased_sentences(model, tokenizer, pattern, num_beams=10, num_return_sequences=10)
        new_patterns += [pattern] + synonyms
    # Update the patterns in the intent
    intent['patterns'] = new_patterns

# Save the updated json
with open('Dataset/updated_mentalhealth.json', 'w') as json_file:
    json.dump(data, json_file)




