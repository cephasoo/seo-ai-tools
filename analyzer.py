import sys
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
# IMPORTANT: Replace this with your actual key from serpapi.com
SERPAPI_KEY = "5ec9ba483195d2e93b2ed08be47ef8d9b81604e200bdc0cadd6388481cc3c944" 

# --- MODEL LOADING (Happens only once when the service starts) ---
print("Loading Gemma-2B Model...", file=sys.stderr)
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded.", file=sys.stderr)

# --- HELPER FUNCTIONS ---
def mean_pooling(hidden_states, attention_mask):
    token_embeddings = hidden_states
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    encoded_input = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True)
        embedding = mean_pooling(model_output.hidden_states[-1], encoded_input['attention_mask'])
    return F.normalize(embedding, p=2, dim=1)[0].tolist()

def scrape_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from main content areas, ignoring nav/footer
        for tag in soup(['nav', 'footer', 'header', 'script', 'style']):
            tag.decompose()
        return ' '.join(soup.stripped_strings)[:4000] # Limit text to avoid huge processing
    except Exception:
        return None

# --- MAIN SCRIPT LOGIC (if this script were to be run directly) ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No query provided"}), file=sys.stderr)
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print(f"Analyzing SERP for query: {query}", file=sys.stderr)
    
    # 1. SCRAPE SERP
    search = GoogleSearch({"q": query, "api_key": SERPAPI_KEY, "num": 15})
    results = search.get_dict().get("organic_results", [])
    
    # 2. SCRAPE & EMBED CONTENT
    scraped_data = []
    for result in results:
        content = scrape_url(result.get("link"))
        if content:
            print(f"Embedding: {result.get('title')}", file=sys.stderr)
            scraped_data.append({
                "title": result.get("title"),
                "source": result.get("source", result.get("displayed_link")),
                "embedding": get_embedding(content)
            })

    # 3. CLUSTER
    if len(scraped_data) < 3:
         print(json.dumps({"error": "Not enough data to perform analysis"}), file=sys.stderr)
         sys.exit(1)
         
    embeddings = [item['embedding'] for item in scraped_data]
    # We'll create 3 clusters to find dominant themes
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(embeddings)
    
    # 4. SUMMARIZE & STRUCTURE OUTPUT
    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        cluster_name = f"Cluster {chr(65 + label)}" # A, B, C
        if cluster_name not in clusters:
            clusters[cluster_name] = {"sources": [], "titles": []}
        clusters[cluster_name]["sources"].append(scraped_data[i]["source"])
        clusters[cluster_name]["titles"].append(scraped_data[i]["title"])

    final_output = {
        "query": query,
        "clusters": [
            {"name": name, "sources": data["sources"], "concepts": data["titles"]} 
            for name, data in clusters.items()
        ],
        "insights": {
            "dominant_intent": "Analysis needed to determine dominant intent.",
            "content_gap": "Further analysis of cluster concepts needed to identify gaps."
        }
    }
    
    # Print final JSON to stdout for n8n
    print(json.dumps(final_output))
