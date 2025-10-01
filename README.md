# SEO AI Tools

This repository contains the source code for a Flask-based API server that provides a suite of AI-powered tools for advanced SEO and content strategy. The system leverages custom fine-tuned Gemma models for embedding and generation, integrates with SerpApi for real-time search data, and includes a Retrieval-Augmented Generation (RAG) pipeline for querying an internal knowledge base.

---

## Features

This API provides several key endpoints for programmatic SEO and content tasks:

* **`/analyze`**: Performs a real-time SERP analysis for a given topic, returning k-means clustered results and rich search features like AI Overviews.
* **`/analyze_url`**: Scrapes, cleans, and generates a structured summary for a single URL.
* **`/generate`**: A flexible endpoint for various generative tasks, such as drafting articles, creating outlines, or revamping content based on detailed instructions.
* **`/embed_and_index`**: Adds new text documents to the RAG knowledge base by generating embeddings and storing them in a FAISS vector store.
* **`/retrieve_context`**: Fetches the most relevant context from the knowledge base for a given query to be used in RAG prompts.

---

## Setup Instructions

Follow these steps to get the API server running locally.

### 1. Prerequisites

* Python 3.9+
* Git

### 2. Clone the Repository

```bash
git clone [https://github.com/cephasoo/seo-ai-tools.git](https://github.com/cephasoo/seo-ai-tools.git)
cd seo-ai-tools
