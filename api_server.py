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
from flask import Flask, request, jsonify
import logging

# --- CONFIGURATION ---
SERPAPI_KEY = "5ec9ba483195d2e93b2ed08be47ef8d9b81604e200bdc0cadd6388481cc3c944" 

# --- MODEL LOADING (Happens only once on startup) ---
print("Loading Gemma-2B Model... This will take time and RAM.", file=sys.stderr)
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded successfully.", file=sys.stderr)

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
        for tag in soup(['nav', 'footer', 'header', 'script', 'style']):
            tag.decompose()
        return ' '.join(soup.stripped_strings)[:4000]
    except Exception:
        return None

# --- FLASK API SETUP ---
app = Flask(__name__)
logging.getLogger('werkzeug').disabled = True

@app.route('/embed', methods=['POST'])
def embed_route():
    try:
        json_data = request.get_json()
        text_input = json_data['text']
        embedding = get_embedding(text_input)
        output = {"text": text_input, "embedding": embedding}
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_route():
    try:
        json_data = request.get_json()
        query = json_data['topic']

        search = GoogleSearch({"q": query, "api_key": SERPAPI_KEY, "num": 15})
        results = search.get_dict().get("organic_results", [])

        scraped_data = []
        for result in results[:10]: # Limit to 10 to speed up
            content = scrape_url(result.get("link"))
            if content:
                scraped_data.append({
                    "title": result.get("title"),
                    "source": result.get("source", result.get("displayed_link")),
                    "embedding": get_embedding(content)
                })

        if len(scraped_data) < 3:
            return jsonify({"error": "Not enough data to perform analysis"})

        embeddings = [item['embedding'] for item in scraped_data]
        kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(embeddings)

        clusters = {}
        for i, label in enumerate(kmeans.labels_):
            cluster_name = f"Cluster {chr(65 + label)}"
            if cluster_name not in clusters:
                clusters[cluster_name] = {"sources": [], "titles": []}
            clusters[cluster_name]["sources"].append(scraped_data[i]["source"])
            clusters[cluster_name]["titles"].append(scraped_data[i]["title"])

        final_output = {
            "query": query,
            "clusters": [{"name": name, "sources": data["sources"], "concepts": data["titles"]} for name, data in clusters.items()]
        }
        return jsonify(final_output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("AI API Server starting on http://0.0.0.0:5001 with /embed and /analyze routes.", file=sys.stderr)
    app.run(host='0.0.0.0', port=5001)
