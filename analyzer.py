import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer # <-- NEW IMPORT
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from config import SERP_API_KEY

# --- MODEL LOADING (HYBRID APPROACH) ---

# 1. Load the large Instruction-Tuned model for GENERATIVE tasks
print("Loading Gemma-2B (Instruction-Tuned) Model for Generation...", file=sys.stderr)
gen_model_name = "google/gemma-2b"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)
print("Generative model loaded.", file=sys.stderr)

# 2. Load the specialized Gemma model for EMBEDDING tasks
print("Loading Gemma_2b_en Embedding Model...", file=sys.stderr)
# Use 'cuda' if you have a GPU, otherwise 'cpu'
embedding_model = SentenceTransformer("google/gemma_2b_en", device='cpu') 
print("Embedding model loaded.", file=sys.stderr)


# --- HELPER FUNCTIONS ---

# REMOVED: The complex mean_pooling function is no longer needed.

# REPLACED: The get_embedding function is now simpler and more powerful
def get_embedding(text):
    """Generates an embedding using the specialized Gemma embedding model."""
    # The model handles tokenization, pooling, and normalization automatically.
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def scrape_url(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['nav', 'footer', 'header', 'script', 'style']):
            tag.decompose()
        return ' '.join(soup.stripped_strings)[:4000]
    except Exception:
        return None

# ENHANCED: This function now correctly uses the large GENERATIVE model
def generate_cluster_synthesis(cluster_content, query):
    """Uses the instruction-tuned generative model to analyze cluster content."""
    prompt_text = f"""
    (T) Task: You are a semantic SEO analyst...
    (C) Context: These texts were grouped together...
    (R) Reference Text:\n---\n{cluster_content}\n---
    (E) Evaluation: Your output MUST be a single, valid JSON object...
    (I) Iteration: Focus only on the shared meaning...
    """
    inputs = gen_tokenizer(prompt_text, return_tensors="pt")
    outputs = gen_model.generate(**inputs, max_new_tokens=200)
    generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        json_str = generated_text[generated_text.find('{'):generated_text.rfind('}')+1]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {"intent": "Analysis Error", "concepts": []}

# --- MAIN SCRIPT LOGIC (Unchanged, but now calls the right functions) ---
if __name__ == "__main__":
    # ... (The rest of your main script logic remains exactly the same)
    # ... It will automatically call the new get_embedding and generate_cluster_synthesis functions.
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No query provided"}), file=sys.stderr)
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Analyzing SERP for query: {query}", file=sys.stderr)

    # 1. SCRAPE SERP
    search = GoogleSearch({"q": query, "api_key": SERP_API_KEY, "num": 15})
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
                "content": content,
                "embedding": get_embedding(content) # Calls the new, efficient function
            })

    # 3. CLUSTER
    if len(scraped_data) < 3:
        print(json.dumps({"error": "Not enough data to perform analysis"}), file=sys.stderr)
        sys.exit(1)

    embeddings = [item['embedding'] for item in scraped_data]
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(embeddings)

    # 4. SUMMARIZE & STRUCTURE ENRICHED OUTPUT
    enriched_clusters = []
    cluster_groups = {i: [] for i in range(3)}
    for i, label in enumerate(kmeans.labels_):
        cluster_groups[label].append(scraped_data[i])

    for i, group_data in cluster_groups.items():
        if not group_data:
            continue
        
        combined_content = " ".join([d['content'] for d in group_data])[:8000]
        print(f"Synthesizing Cluster {chr(65 + i)}...", file=sys.stderr)
        synthesis = generate_cluster_synthesis(combined_content, query) # Calls the generative function

        enriched_clusters.append({
            "name": f"Cluster {chr(65 + i)}: {synthesis.get('intent', 'N/A')}",
            "intent": synthesis.get('intent', 'N/A'),
            "sources": [d['source'] for d in group_data],
            "concepts": synthesis.get('concepts', [])
        })

    final_output = {
        "query": query,
        "clusters": enriched_clusters,
        "insights": {
            "dominant_intent": "Analysis to be completed by the downstream Generative AI Text Engine.",
            "content_gap": "Analysis to be completed by the downstream Generative AI Text Engine."
        }
    }
    
    print(json.dumps(final_output))
