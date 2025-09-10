import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the pooling function to create a single embedding from token vectors
def mean_pooling(hidden_states, attention_mask):
    token_embeddings = hidden_states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load the pre-downloaded model and tokenizer from the cache
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Read the text input that will be passed from the n8n Execute Command node
text_input = sys.stdin.read()
sentences = [text_input]

# Main embedding generation logic
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input, output_hidden_states=True)
    sentence_embeddings = mean_pooling(model_output.hidden_states[-1], encoded_input['attention_mask'])

# Normalize the embeddings for consistent similarity scores
normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Prepare the output as a clean JSON object
# This is the data that n8n will receive in the 'stdout' property
output = {
    "text": text_input,
    "embedding": normalized_embeddings[0].tolist()
}
# Print the JSON to standard output
print(json.dumps(output))
