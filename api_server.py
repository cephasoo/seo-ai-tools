# api_server.py (v8.3 - Consolidated SERP, JSON Parsing & Robust Rich Features - FIXED)

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
    """Uses ScraperAPI to reliably fetch and render page content with cost control and improved filtering."""
    try:
        # 1. Clean URL and get dynamic context
        clean_url = url.replace('[', '').replace(']', '').strip()
        country_code = data.get('target_country', 'us')

        # 2. Build Premium Payload with Cost Control (50 credits max per scrape attempt)
        # 🐛 FIX: The 'payload' dictionary definition was missing, causing the NameError.
        payload = {
            'api_key': SCRAPER_API_KEY,
            'url': clean_url,
            'country_code': country_code,
            'render': 'true', # Best practice for modern, JavaScript-heavy sites
            'premium': 'true', # Use a premium proxy for better reliability
            'max_cost': '50', # CRITICAL COST GUARDRAIL: Stops credit bleed
            'session_number': '1' # Use a sticky session for better results
        }

        # 3. Send request with a generous, but not infinite, timeout
        response = requests.get(SCRAPER_API_ENDPOINT, params=payload, timeout=60)
        
        # --- CRITICAL REFINEMENT: Check status before parsing ---
        if response.status_code != 200:
            print(f"ScraperAPI returned non-200 status {response.status_code} for {url}", file=sys.stderr)
            response.raise_for_status()

        # 4. Anti-bot/Error Check 
        if not response.text or "you've been blocked" in response.text.lower() or "checking if the site connection is secure" in response.text.lower() or "access is restricted" in response.text.lower():
            print(f"ScraperAPI returned a block page or restricted access for {url}", file=sys.stderr)
            return None

        # 5. Parse and clean content (Improved cleanup for scholarly/paywall sites)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Added more aggressive tag decomposition, especially for scholarly/paywall headers
        for tag in soup(['nav', 'footer', 'header', 'script', 'style', 'aside', 'form', 'iframe', 'canvas', 'svg']):
            tag.decompose()

        content = ' '.join(soup.stripped_strings)

        # --- FINAL QUALITY GATE ---
        # If content is still too short after cleaning, filter it out.
        if len(content) < 500:
            print(f"Content for {url} too short after cleaning ({len(content)} chars). Filtering.", file=sys.stderr)
            return None

        return content[:5000]

    except requests.RequestException as e:
        # This catches the 500 error from the log gracefully
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
        # 1. Isolate the RAW JSON string block (from the first { to the last })
        json_match = re.search(r'\{[\s\S]*?\}', generated_text)
        if not json_match:
            raise ValueError("No complete JSON object found in model output.")
        json_str = json_match.group(0)

        # 2. Aggressive Parsing using ast.literal_eval (Robust to single quotes/newlines)
        clean_json_str = json_str.strip().replace('\n', ' ').replace('\t', ' ')
        data_dict = ast.literal_eval(clean_json_str)

        return data_dict

    except (SyntaxError, ValueError, IndexError) as e:
        print(f"Failed to decode JSON from model output in cluster synthesis. Error: {e}", file=sys.stderr)
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
        # 1. Isolate the RAW JSON string block
        json_match = re.search(r'\{[\s\S]*?\}', generated_text)
        if not json_match:
            raise ValueError("No complete JSON object found in model output.")
        json_str = json_match.group(0)

        # 2. Aggressive Parsing using ast.literal_eval
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

# 1. SERP Analysis Endpoint (Topic-Driven Research - CONSOLIDATED & ROBUST)
@app.route('/analyze', methods=['POST'])
def analyze_route():
    data = request.json
    query = data.get('topic')
    channel = data.get('channel')
    timestamp = data.get('ts')

    if not query: return jsonify({"error": "No topic/query provided"}), 400
    print(f"Analyzing SERP for query: {query} using CONSOLIDATED 'google' engine.", file=sys.stderr)

    # --- 1. CONSOLIDATED CALL FOR RICH & ORGANIC RESULTS (Standard Google Engine) ---
    consolidated_search_params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "num": 10 # Standard engine limit for primary organic results
    }

    try:
        results_data = GoogleSearch(consolidated_search_params).get_dict()
    except Exception as e:
        print(f"SerpApi 'google' engine call failed: {e}", file=sys.stderr)
        return jsonify({"error": "SerpApi call failed during consolidated search.", "details": str(e)}), 500

    organic_results = results_data.get("organic_results", [])

    # --- CONDITIONAL CALL FOR FULL AI OVERVIEW (SECONDARY CALL - REQUIRED FOR FULL AIO CONTENT) ---
    rich_features = {}
    ai_overview_data = results_data.get('ai_overview')

    # CRITICAL: Logic to handle lazy-loaded AIO which requires a token follow-up
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
            try:
                aio_response = GoogleSearch(aio_params).get_dict()
                if aio_response.get('ai_overview'):
                    # Success: Use the full response from the dedicated AIO engine
                    rich_features['ai_overview'] = aio_response['ai_overview']
                else:
                    # Failure in secondary call: Fall back to the initial, likely partial, data
                    rich_features['ai_overview'] = ai_overview_data
            except Exception as e:
                # Handle total failure of secondary call gracefully
                print(f"Secondary AIO call failed: {e}", file=sys.stderr)
                rich_features['ai_overview'] = ai_overview_data
        else:
            # Full AIO was in the primary response (Rare but possible)
            rich_features['ai_overview'] = ai_overview_data

    # Capture all high-value rich elements from the consolidated result
    for key in [
        'knowledge_graph',
        'featured_snippet',
        'answer_box',
        'related_questions',
        'related_searches',
        'top_stories',         # New
        'local_results',       # New
        'videos',              # New
        'shopping_results',    # New
        'inline_tweets'        # New
    ]:
        if key in results_data:
            # Only add if not already populated by the AIO logic above
            if key not in rich_features:
                rich_features[key] = results_data[key]


    # --- 2. SCRAPING AND CLUSTERING (Now using up to 10 organic results) ---
    scraped_data = []

    for result in organic_results:
        link = result.get("link")
        content = scrape_url(link, data)

        # Check if content is garbage (error page, too short, or explicitly blocked)
        if content and len(content) > 500 and "access denied" not in content.lower():
            scraped_data.append({
                "title": result.get("title"), "source": result.get("displayed_link", result.get("source")),
                "content": content, "embedding": get_embedding(content)
            })
        else:
            print(f"Filtered out low-quality/error content from {link}", file=sys.stderr)


    if len(scraped_data) < 3: return jsonify({"error": "Not enough **clean** data to perform SERP analysis (Need 3+, found: " + str(len(scraped_data)) + ")", "results_found": len(scraped_data)}), 400

    # K-Means Clustering Logic
    embeddings = [item['embedding'] for item in scraped_data]
    num_clusters = min(4, len(scraped_data))
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

    # 3. Final Output - Returns clusters AND rich features
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
