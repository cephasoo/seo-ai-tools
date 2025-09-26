# api_server.py (v8.2 - Triple-Call Synthesis with Data Filtering & Cost Control)

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
from config import SERP_API_KEY, SCRAPER_API_KEY 
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
    """Uses ScraperAPI to reliably fetch and render page content with cost control."""
    try:
        # 1. Clean URL and get dynamic context
        clean_url = url.replace('[', '').replace(']', '').strip()
        country_code = data.get('target_country', 'us') 
        
        # 2. Build Premium Payload with Cost Control (50 credits max per scrape attempt)
        payload = {
            'api_key': SCRAPER_API_KEY,
            'url': clean_url,
            'render': 'true', 
            'premium': 'true',          # High-reliability proxy pool
            'country_code': country_code,
            'max_cost': '50'            # CRITICAL COST GUARDRAIL: Stops credit bleed
        }
        
        # 3. Send request with a generous, but not infinite, timeout
        response = requests.get(SCRAPER_API_ENDPOINT, params=payload, timeout=60)
        response.raise_for_status() 

        # 4. Anti-bot/Error Check (ScraperAPI passes the error page as 200/400 often)
        if not response.text or "you've been blocked" in response.text.lower() or "checking if the site connection is secure" in response.text.lower():
             print(f"ScraperAPI returned a block page for {url}", file=sys.stderr)
             return None

        # 5. Parse and clean content
        soup = BeautifulSoup(response.text, 'html.parser')
        
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
        
        # Aggressive cleaning for structural validity
        json_str = json_str.strip().replace("'", '"').replace('\n', '\\n').replace('\t', '\\t') 
        
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
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
        json_match = re.search(r'\{[\s\S]*?\}', generated_text) 
        if not json_match:
            raise ValueError("No complete JSON object found in model output.")
        json_str = json_match.group(0)

        # 2. Aggressive Parsing using ast.literal_eval (The most resilient method)
        # Clean newlines/tabs before parsing
        clean_json_str = json_str.strip().replace('\n', ' ').replace('\t', ' ')
        data_dict = ast.literal_eval(clean_json_str) 

        return data_dict
        
    except (SyntaxError, ValueError, IndexError) as e:
        print(f"Failed to decode JSON from model output (FINAL FALLBACK). Error: {e}", file=sys.stderr)
        return {
            "title": "Synthesis Error",
            "summary": "Failed to parse detailed summary from AI model.",
            "key_arguments": [f"Parsing failed due to: {e}"],
            "concepts": ["Error"]
        }


# --- API ENDPOINTS ---

# 1. SERP Analysis Endpoint (Topic-Driven Research - TRIPLE CALL)
@app.route('/analyze', methods=['POST'])
def analyze_route():
    data = request.json
    query = data.get('topic')
    channel = data.get('channel')
    timestamp = data.get('ts')

    if not query: return jsonify({"error": "No topic/query provided"}), 400
    print(f"Analyzing SERP for query: {query} using DUAL-CALL strategy.", file=sys.stderr)
    
    # --- 1. CALL FOR RICH RESULTS (Standard Engine, Shallow Search) ---
    rich_search_params = {
        "engine": "google", 
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 10 
    }
    rich_results_data = GoogleSearch(rich_search_params).get_dict()
    
    # --- CONDITIONAL CALL FOR FULL AI OVERVIEW (3rd CALL) ---
    rich_features = {}
    ai_overview_data = rich_results_data.get('ai_overview')

    if ai_overview_data:
        # Check if the AIO is lazy-loaded and requires a token follow-up
        if ai_overview_data.get('page_token'):
            print("AIO token found. Making secondary call to fetch full AIO content.", file=sys.stderr)
            token = ai_overview_data['page_token']
            aio_params = {
                "engine": "google_ai_overview",
                "page_token": token,
                "api_key": SERP_API_KEY
            }
            aio_response = GoogleSearch(aio_params).get_dict()
            
            if aio_response.get('ai_overview'):
                rich_features['ai_overview'] = aio_response['ai_overview']
            else:
                rich_features['ai_overview'] = ai_overview_data
        else:
            # Full AIO was in the primary response
            rich_features['ai_overview'] = ai_overview_data

    # Capture other rich elements 
    for key in ['knowledge_graph', 'featured_snippet', 'answer_box']:
        if key in rich_results_data:
            rich_features[key] = rich_results_data[key]


    # --- 2. CALL FOR DEEP ORGANIC RESULTS (Fast & Light Engine) ---
    deep_search_params = {
        "engine": "google_light_fast", 
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 44 # Maximum organic depth
    }
    deep_results_data = GoogleSearch(deep_search_params).get_dict()
    organic_results = deep_results_data.get("organic_results", [])
    
    # --- 3. SCRAPING AND CLUSTERING (Using 100 organic results) ---
    scraped_data = []
    
    for result in organic_results:
        link = result.get("link")
        # STRATEGIC DATA FILTERING (Pillar A Decommission Replacement)
        content = scrape_url(link, data) 
        
        # Check if content is garbage (error page, too short, or explicitly blocked)
        if content and len(content) > 500 and "access denied" not in content.lower():
            scraped_data.append({
                "title": result.get("title"), "source": result.get("displayed_link", result.get("source")),
                "content": content, "embedding": get_embedding(content)
            })
        else:
            print(f"Filtered out low-quality/error content from {link}", file=sys.stderr)


    if len(scraped_data) < 3: return jsonify({"error": "Not enough **clean** data to perform SERP analysis", "results_found": len(scraped_data)}), 400
    
    # K-Means Clustering Logic
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
        synthesis = generate_cluster_synthesis(combined_content, query)
        enriched_clusters.append({
            "name": f"Cluster {chr(65 + i)}: {synthesis.get('intent', 'N/A')}",
            "intent": synthesis.get('intent', 'N/A'),
            "sources": [d['source'] for d in group_data],
            "concepts": synthesis.get('concepts', [])
        })

    # 4. Final Output - Returns clusters AND rich features
    final_output = {
        "channel": channel,
        "ts": timestamp,
        "payload": {
            "query": query,
            "clusters": enriched_clusters, 
            "rich_features": rich_features # PASSES AI OVERVIEW AND SNIPPETS
        }
    }

    return jsonify(final_output)


# 2. On-Page Analysis Endpoint (URL-Driven Research)
@app.route('/analyze_url', methods=['POST'])
def analyze_url_route():
    data = request.json
    url = data.get('url') 
    query = data.get('topic')
    channel = data.get('channel')
    timestamp = data.get('ts')

    if not url: return jsonify({"error": "No URL provided for analysis"}), 400
    
    print(f"Analyzing single URL: {url}", file=sys.stderr)
    
    content = scrape_url(url, data) 
    
    if not content: 
        return jsonify({"error": "Failed to retrieve content from URL"}), 400
    
    # 2. Generate detailed summary (Uses enhanced summary helper)
    summary_data = generate_url_summary(content, query) 
    
    # 3. Final Output - Returns rich summary data
    final_output = {
        "channel": channel,
        "ts": timestamp,
        "url": url, 
        "payload": summary_data
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
