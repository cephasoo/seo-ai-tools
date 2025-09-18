# api_server.py (v6.4 - Final Fixes & Context Persistence)
import sys
import torch
import json
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from config import SERP_API_KEY
import urllib3

# Suppress the InsecureRequestWarning from the scraper.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
app = Flask(__name__)


# --- MODEL LOADING (UNIFIED) ---
print("Loading Custom Fine-Tuned SEO Embedding Model...", file=sys.stderr)
embedding_model_path = 'my-expert-seo-embedder'
embedding_model = SentenceTransformer(embedding_model_path, device='cpu')
print("Custom embedding model loaded.", file=sys.stderr)

print("Loading Merged Fine-Tuned Generative Model (SEO Co-pilot)...", file=sys.stderr)
gen_model_path = 'my-expert-seo-generator-merged'
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_path)
print("Unified generative model loaded.", file=sys.stderr)


# --- HELPER FUNCTIONS ---
def get_embedding(text):
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def scrape_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['nav', 'footer', 'header', 'script', 'style', 'aside', 'form']):
            tag.decompose()
        content = ' '.join(soup.stripped_strings)
        if "you've been blocked" in content.lower() or "checking if the site connection is secure" in content.lower():
            return None
        return content[:5000]
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}", file=sys.stderr)
        return None

def generate_cluster_synthesis(cluster_content, query):
    prompt_text = f"""
Analyze the Reference Text, which contains content scraped from search results for the query '{query}'.
Your output MUST be a single, valid JSON object with two keys:
1. "intent": A short phrase describing the theme.
2. "concepts": An array of 2-3 key strings representing specific topics.
Reference Text:
---
{cluster_content}
---
JSON Output:
"""
    inputs = gen_tokenizer(prompt_text, return_tensors="pt")
    outputs = gen_model.generate(**inputs, max_new_tokens=200)
    generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        json_str = generated_text[generated_text.find('{'):generated_text.rfind('}')+1]
        if not json_str: raise ValueError("No JSON object found")
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to decode JSON from model output. Raw text: '{generated_text}'. Error: {e}", file=sys.stderr)
        return {"intent": "Analysis Error", "concepts": []}

# --- API ENDPOINTS ---
@app.route('/analyze', methods=['POST'])
def analyze_route():
    data = request.json
    query = data.get('topic')
    
    # --- START OF MODIFIED CODE: Extract and pass through context ---
    channel = data.get('channel')
    timestamp = data.get('ts')
    # --- END OF MODIFIED CODE ---

    if not query: return jsonify({"error": "No topic/query provided"}), 400
    print(f"Analyzing SERP for query: {query}", file=sys.stderr)
    search = GoogleSearch({"q": query, "api_key": SERP_API_KEY, "num": 15})
    results = search.get_dict().get("organic_results", [])
    scraped_data = []
    for result in results:
        content = scrape_url(result.get("link"))
        if content:
            scraped_data.append({
                "title": result.get("title"), "source": result.get("source", result.get("displayed_link")),
                "content": content, "embedding": get_embedding(content)
            })
    if len(scraped_data) < 3: return jsonify({"error": "Not enough data to perform analysis", "results_found": len(scraped_data)})
    embeddings = [item['embedding'] for item in scraped_data]
    num_clusters = min(3, len(scraped_data))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(embeddings)
    enriched_clusters = []
    cluster_groups = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(kmeans.labels_):
        cluster_groups[label].append(scraped_data[i])
    for i, group_data in cluster_groups.items():
        if not group_data: continue
        combined_content = " ".join([d['content'] for d in group_data])[:8000]
        print(f"Synthesizing Cluster {chr(65 + i)}...", file=sys.stderr)
        synthesis = generate_cluster_synthesis(combined_content, query)
        enriched_clusters.append({
            "name": f"Cluster {chr(65 + i)}: {synthesis.get('intent', 'N/A')}",
            "intent": synthesis.get('intent', 'N/A'),
            "sources": [d['source'] for d in group_data],
            "concepts": synthesis.get('concepts', [])
        })
    
    # --- START OF MODIFIED CODE: Add context to final output ---
    final_output = {
        "channel": channel,
        "ts": timestamp,
        "payload": {
            "query": query,
            "clusters": enriched_clusters
        }
    }
    # --- END OF MODIFIED CODE ---
    
    return jsonify(final_output)

@app.route('/generate', methods=['POST'])
def generate_route():
    data = request.json
    instruction = data.get('instruction')
    user_input = data.get('input', '')

    # --- START OF MODIFIED CODE: Extract and pass through context ---
    channel = data.get('channel')
    timestamp = data.get('ts')
    # --- END OF MODIFIED CODE ---

    if not instruction: return jsonify({"error": "Instruction is required"}), 400
    prompt = f"### Instruction: {instruction}\n### Input: {user_input}\n### Output:"
    inputs = gen_tokenizer(prompt, return_tensors="pt")
    print("Generating specialized response...", file=sys.stderr)
    outputs = gen_model.generate(**inputs, max_new_tokens=500)
    result_full = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    result_only_output = result_full.split("### Output:")[1].strip()

    # --- START OF MODIFIED CODE: Add context to final output ---
    final_output = {
        "channel": channel,
        "ts": timestamp,
        "payload": {
            "response": result_only_output
        }
    }
    # --- END OF MODIFIED CODE ---
    
    return jsonify(final_output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
