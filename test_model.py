import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# CORRECTED Mean Pooling Function
def mean_pooling(hidden_states, attention_mask):
    # The 'hidden_states' is the tensor of all token embeddings.
    # We no longer incorrectly access the first element.
    token_embeddings = hidden_states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want to encode
sentences = ["This is the first sentence.", "This is the second sentence for our test."]

print("Loading Gemma-2B Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

print("Loading Gemma-2B Model... (This will be much faster as it uses the cache)")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
print("Model loaded successfully.")

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
print("Generating embeddings...")
with torch.no_grad():
    model_output = model(**encoded_input, output_hidden_states=True)
    # Pass the last hidden state directly to our corrected pooling function
    sentence_embeddings = mean_pooling(model_output.hidden_states[-1], encoded_input['attention_mask'])

# Normalize embeddings for better similarity comparison
normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("\nSuccessfully generated embeddings with Gemma-2B!")
print("Your environment is ready for state-of-the-art semantic analysis.")
print("\nVector for the first sentence (first 5 dimensions):", normalized_embeddings[0][:5])
