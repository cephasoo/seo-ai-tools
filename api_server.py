# api_server.py (v7.1 - Dual Analysis & ScraperAPI Integration)

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
# IMPORTANT: Ensure your config.py has both keys
from config import SERP_API_KEY, SCRAPER_API_KEY 
import urllib3

# Suppress the InsecureRequestWarning if using verify=False (though ScraperAPI usually handles SSL)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
app = Flask(__name__)
# Define the ScraperAPI Endpoint
SCRAPER_API_ENDPOINT = "http://api.scraperapi.com/"


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

# NOTE: This function requires the 'data' dict to safely access 'target_country'
def scrape_url(url, data): 
    """Uses ScraperAPI to reliably fetch and render page content."""
    try:
        # Get country code from the request data, defaulting to 'us' if absent
        country_code = data.get('target_country', 'us') 
        
        # 1. Define payload for ScraperAPI
        payload = {
            'api_key': SCRAPER_API_KEY,
            'url': url,
            'render': 'true', # CRITICAL: Enables JavaScript rendering
            'country_code': country_code # Dynamic Geotargeting
        }
        
        # 2. Send the request to the ScraperAPI endpoint
        response = requests.get(SCRAPER_API_ENDPOINT, params=payload, timeout=60)
        response.raise_for_status() 

        if not response.text or "you've been blocked" in response.text.lower():
             print(f"ScraperAPI returned a block page for {url}", file=sys.stderr)
             return None

        # 3. Parse the successfully retrieved HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Your existing content cleaning logic 
        for tag in soup(['nav', 'footer', 'header', 'script', 'style', 'aside', 'form']):
            tag.decompose()
        content = ' '.join(soup.stripped_strings)
        
        return content[:5000]
        
    except requests.RequestException as e:
        print(f"Error scraping {url} via ScraperAPI: {e}", file=sys.stderr)
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

# 1. SERP Analysis Endpoint (Topic-Driven Research)
@app.route('/analyze', methods=['POST'])
def analyze_route():
    data = request.json
    query = data.get('topic')
    channel = data.get('channel')
    timestamp = data.get('ts')

    if not query: return jsonify({"error": "No topic/query provided"}), 400
    print(f"Analyzing SERP for query: {query}", file=sys.stderr)
    
    # 1. Get SERP results (via SerpAPI)
    search = GoogleSearch({"q": query, "api_key": SERP_API_KEY, "num": 15})
    results = search.get_dict().get("organic_results", [])
    
    scraped_data = []
    
    # 2. Iterate and scrape (Now using ScraperAPI via scrape_url)
    for result in results:
        # FIX: Pass the 'data' context into scrape_url
        content = scrape_url(result.get("link"), data) 
        if content:
            scraped_data.append({
                "title": result.get("title"), "source": result.get("displayed_link", result.get("source")),
                "content": content, "embedding": get_embedding(content)
            })
    
    if len(scraped_data) < 3: return jsonify({"error": "Not enough data to perform SERP analysis", "results_found": len(scraped_data)}), 400
    
    # 3. Clustering and Synthesis
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

    # 4. Final Output
    final_output = {
        "channel": channel,
        "ts": timestamp,
        "payload": {
            "query": query,
            "clusters": enriched_clusters
        }
    }

    return jsonify(final_output)


# 2. On-Page Analysis Endpoint (URL-Driven Research)
# NOTE: You will need to implement generate_summary_from_content 
# or adapt generate_cluster_synthesis for single article summarization.
@app.route('/analyze_url', methods=['POST'])
def analyze_url_route():
    data = request.json
    url = data.get('url') # N8N passes one URL here per iteration
    query = data.get('topic')
    channel = data.get('channel')
    timestamp = data.get('ts')

    if not url: return jsonify({"error": "No URL provided for analysis"}), 400
    
    print(f"Analyzing single URL: {url}", file=sys.stderr)
    
    # 1. Scrape only the requested URL, using the robust scraper
    content = scrape_url(url, data) 
    
    if not content: 
        return jsonify({"error": "Failed to retrieve content from URL"}), 400
    
    # 2. Use the synthesis model to summarize the single page's content.
    # We will temporarily use the clustering synthesis function, but ideally 
    # you would create a new one optimized for summarization.
    synthesis_result = generate_cluster_synthesis(content, query) 
    
    # 3. Return the structured result for N8N to merge.
    final_output = {
        "url": url, 
        "payload": synthesis_result
    }
    
    return jsonify(final_output)


# 3. Generative Task Endpoint
@app.route('/generate', methods=['POST'])
def generate_route():
    data = request.json
    instruction = data.get('instruction')
    user_input = data.get('input', '')

    channel = data.get('channel')
    timestamp = data.get('ts')

    if not instruction: return jsonify({"error": "Instruction is required"}), 400
    
    prompt = f"### Instruction: {instruction}\n### Input: {user_input}\n### Output:"
    inputs = gen_tokenizer(prompt, return_tensors="pt")
    
    print("Generating specialized response...", file=sys.stderr)
    outputs = gen_model.generate(**inputs, max_new_tokens=500)
    result_full = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Simple extraction of the output segment
    result_only_output = result_full.split("### Output:")[1].strip()

    final_output = {
        "channel": channel,
        "ts": timestamp,
        "payload": {
            "response": result_only_output
        }
    }

    return jsonify(final_output)

if __name__ == '__main__':
    # Running on all addresses allows external access from N8N
    app.run(host='0.0.0.0', port=5001)
