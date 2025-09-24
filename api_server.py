# api_server.py (v7.3 - Dual Analysis with Enhanced On-Page Output)

import sys
import torch
import json
import requests
import re
import ast
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from config import SERP_API_KEY, SCRAPER_API_KEY # Ensure config.py is updated
import urllib3

# Suppress the InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
app = Flask(__name__)
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

def scrape_url(url, data): 
    """Uses ScraperAPI to reliably fetch and render page content."""
    try:
        # FIX: Ensure the API receives a clean URL by stripping internal brackets
        clean_url = url.replace('[', '').replace(']', '').strip()
        country_code = data.get('target_country', 'us') 
        
        payload = {
            'api_key': SCRAPER_API_KEY,
            'url': clean_url,
            'render': 'true', 
            'country_code': country_code
        }
        
        response = requests.get(SCRAPER_API_ENDPOINT, params=payload, timeout=60)
        response.raise_for_status() 

        if not response.text or "you've been blocked" in response.text.lower():
             print(f"ScraperAPI returned a block page for {url}", file=sys.stderr)
             return None

        # Parse the successfully retrieved HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Content cleaning
        for tag in soup(['nav', 'footer', 'header', 'script', 'style', 'aside', 'form']):
            tag.decompose()
        content = ' '.join(soup.stripped_strings)
        
        return content[:5000]
        
    except requests.RequestException as e:
        print(f"Error scraping {url} via ScraperAPI: {e}", file=sys.stderr)
        return None

# Helper 1: Used for K-Means Synthesis (Topic/Intent/Concepts only)
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
        # Robust JSON extraction and cleaning
        json_str = generated_text[generated_text.find('{'):generated_text.rfind('}')+1]
        
        if not json_str: 
            raise ValueError("No JSON object found in model output.")
        
        json_str = json_str.strip().replace("'", '"') # Aggressive quote and whitespace clean
        
        return json.loads(json_str)
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to decode JSON from model output. Raw text: '{generated_text}'. Error: {e}", file=sys.stderr)
        return {"intent": "Analysis Error", "concepts": []}


# Helper 2: Used for Single URL Detailed Summary (Substance for downstream AI)
def generate_url_summary(url_content, query):
    prompt_text = f"""
Analyze the Reference Article content for the query '{query}'. Provide a detailed summary and extract the key arguments for comparison.
Your output MUST be a single, valid JSON object with four keys:
1. "title": The title of the article.
2. "summary": A concise summary of the article's main point (max 160 characters).
3. "key_arguments": An array of 3-4 bulleted strings representing the core takeaways/arguments.
4. "concepts": An array of 2-3 strings representing the main SEO concepts discussed.
Reference Article Content:
---
{url_content}
---
JSON Output:
"""
    inputs = gen_tokenizer(prompt_text, return_tensors="pt")
    outputs = gen_model.generate(**inputs, max_new_tokens=500)
    generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        # 1. Isolate the RAW JSON string block (from the first { to the last })
        json_str = generated_text[generated_text.find('{'):generated_text.rfind('}')+1]
        
        if not json_str: 
            raise ValueError("No JSON object found in model output.")
        
        # 2. **Aggressive Cleaning for malformed strings/quotes**
        json_str = json_str.strip().replace("'", '"')
        
        # 3. **SAFEST PARSING ATTEMPT (using Python's JSON parser):**
        # Attempt to load the JSON. If this fails, we resort to Regex extraction.
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass # Proceed to aggressive regex reconstruction
            
        # --- AGGRESSIVE REGEX RECONSTRUCTION ---
        # This bypasses structural errors (like unmatched brackets) by finding valid fields.
        
        # Define the fields we are looking for (based on the expected output schema)
        fields = ["title", "summary", "key_arguments", "concepts"]
        reconstructed_data = {}
        
        # Regex to find: "key": [value] or "key": "value"
        for field in fields:
            # Pattern looks for "field_name" followed by a colon and then captures the value.
            # Captures strings ("...") or arrays ([...])
            pattern = rf'"{field}"\s*:\s*(?:"((?:\\.|[^"])*)"|(\[[\s\S]*?\]))'
            match = re.search(pattern, json_str, re.IGNORECASE)
            
            if match:
                value_str = match.group(1) if match.group(1) is not None else match.group(2)
                
                # Try to evaluate the value (handles string vs array types)
                try:
                    # Use literal_eval to handle single-quoted list/array contents
                    reconstructed_data[field] = ast.literal_eval(value_str.strip())
                except (ValueError, SyntaxError):
                    # If evaluation fails, treat it as a raw string (e.g., if it's a quote-free string)
                    reconstructed_data[field] = value_str.strip()

        # If we successfully extracted any data, return the reconstructed dictionary
        if reconstructed_data:
            return reconstructed_data
        else:
            raise ValueError("Reconstruction failed: No key fields found.")
        
    except (ValueError, SyntaxError) as e:
        # Fallback for catastrophic failure
        print(f"Failed to decode JSON from model output (FINAL FALLBACK). Error: {e}", file=sys.stderr)
        return {
            "title": "Synthesis Error",
            "summary": "Failed to parse detailed summary from AI model.",
            "key_arguments": [f"Parsing failed due to: {e}"],
            "concepts": ["Error"]
        }

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
        content = scrape_url(result.get("link"), data) 
        if content:
            scraped_data.append({
                "title": result.get("title"), "source": result.get("displayed_link", result.get("source")),
                "content": content, "embedding": get_embedding(content)
            })
    
    if len(scraped_data) < 3: return jsonify({"error": "Not enough data to perform SERP analysis", "results_found": len(scraped_data)}), 400
    
    # 3. Clustering and Synthesis (Uses cluster synthesis helper)
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

    # 4. Final Output - Returns 'clusters' for SERP data
    final_output = {
        "channel": channel,
        "ts": timestamp,
        "payload": {
            "query": query,
            "clusters": enriched_clusters # SERP analysis returns clusters
        }
    }

    return jsonify(final_output)


# 2. On-Page Analysis Endpoint (URL-Driven Research)
@app.route('/analyze_url', methods=['POST'])
def analyze_url_route():
    data = request.json
    url = data.get('url') # N8N passes one URL here per iteration
    query = data.get('topic')
    channel = data.get('channel')
    timestamp = data.get('ts')

    if not url: return jsonify({"error": "No URL provided for analysis"}), 400
    
    print(f"Analyzing single URL: {url}", file=sys.stderr)
    
    # 1. Scrape only the requested URL
    content = scrape_url(url, data) 
    
    if not content: 
        # API returns a 400 when scraping fails, as defined by the requirement
        return jsonify({"error": "Failed to retrieve content from URL"}), 400
    
    # 2. Generate detailed summary (Uses enhanced summary helper)
    summary_data = generate_url_summary(content, query) 
    
    # 3. Final Output - Returns rich summary data
    final_output = {
        "channel": channel,
        "ts": timestamp,
        "url": url, 
        "payload": summary_data # URL analysis returns rich metadata
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
    app.run(host='0.0.0.0', port=5001)
