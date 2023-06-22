import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

# Define the fine-tuning dataset
class IntentDataset(Dataset):
    def __init__(self, intents, tokenizer):
        self.intents = intents
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.intents)
    
    def __getitem__(self, index):
        intent = self.intents[index]
        patterns = intent['patterns']
        tag = intent['tag']
        
        encoded_patterns = [self.tokenizer(pattern, truncation=True, padding='max_length', max_length=128, return_tensors='pt') for pattern in patterns]
        
        return {
            'input_ids': torch.cat([encoded_pattern['input_ids'] for encoded_pattern in encoded_patterns]),
            'attention_mask': torch.cat([encoded_pattern['attention_mask'] for encoded_pattern in encoded_patterns]),
            'tag': tag
        }

# Define the fine-tuning model
class IntentClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(IntentClassifier, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.classification_layer = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Take the hidden state of the first token [CLS]
        logits = self.classification_layer(pooled_output)
        return logits

# Fine-tuning parameters
path = "jianghc/medical_chatbot"  # Update with the desired model name or local path
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(path)
data_path = "/Users/sarrabenyahia/Documents/GitHub/chatbot-mental-health/Dataset/cleaned_mentalhealth.json"

# Load intents data from JSON file
with open(data_path, 'r') as file:
    intents_data = json.load(file)

# Extract intents from data
intents = intents_data["intents"]

# Hyperparameters for fine-tuning
num_labels = len(intents)
batch_size = 4
learning_rate = 1e-5
num_epochs = 5

# Create the dataset and dataloader
dataset = IntentDataset(intents, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the fine-tuning model
model = IntentClassifier(path, num_labels).to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Fine-tuning loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].squeeze().to(device)
        attention_mask = batch['attention_mask'].squeeze().to(device)
        labels = batch['tag'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
