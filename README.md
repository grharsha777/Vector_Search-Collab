****ARCHITECTRON-X****: Vector Search & RAG on Flipkart Product Reviews
ARCHITECTRON-X is an enterprise-grade semantic search and retrieval‑augmented generation (RAG) pipeline built on top of Flipkart product reviews.

It combines dense vector search (Mistral embeddings + FAISS) with an LLM-based RAG layer to answer product questions using real customer feedback.

Features
End-to-end semantic search pipeline on 200K+ Flipkart reviews (cleaned to ~180K for EDA, ~20K for indexing in fast mode).

Dual embedding strategy: cloud Mistral embeddings (mistral-embed, 1024‑dim) with automatic TF‑IDF + SVD local fallback.

FAISS IndexFlatIP vector index with cosine similarity over L2‑normalised embeddings.

Production-style Mistral client with:

SQLite + in‑memory caching

Automatic retries and graceful degradation to local embeddings.

Robust text preprocessing (URL/HTML/emoji/special-char removal, length control, rich document construction from product name + summary + review).

Semantic search engine with:

Top‑K retrieval

Optional sentiment and minimum rating filters

Pretty CLI-style result printing.

RAG pipeline that:

Retrieves top‑K reviews

Builds a compact, structured context

Calls a Mistral chat model (mistral-small-latest) with a domain‑specific system prompt.

Comprehensive evaluation:

Precision@K, Recall@K, F1@K, NDCG@K, Mean Reciprocal Rank (MRR) with unit‑tested implementations.

Extensive EDA and visualisation suite (rating and sentiment distributions, word clouds, product/price stats, PCA of embedding space).

Architecture
High-level flow:

Raw CSV -> Preprocessing -> Embeddings (Mistral or TF‑IDF+SVD) -> FAISS IndexFlatIP
|
User Query -> Query Embedding -> Top-K Retrieval -> RAG Answer (Mistral LLM)

Main Components
Environment & Config

Central Config dataclass controls dataset paths, sample size, embedding settings, FAISS index path, and evaluation Ks.

Automatic ZIP path discovery for the Flipkart dataset and environment‑driven MISTRAL_API_KEY.

Mistral Client Wrapper

Thin wrapper around the Mistral API with:

Persistent SQLite cache (mistral_cache.sqlite)

TTL for cache invalidation

In‑memory memoisation

Fallback to local embedding backend when no API key is set.

Data Loading & Validation

Reads Flipkart reviews from a ZIP archive (Dataset-SA.csv).

Validates schema, reports shape, null counts, and sample rows.

EDA & Visualisations

Cleans and casts Rate and product_price to numeric; adds review_len and word_count.

Key figures (all saved as PNG):

Rating distribution + sentiment donut + review length violin.

Positive vs negative review word clouds (if wordcloud is available).

Top 15 most-reviewed products and log‑scaled price distribution.

PCA projection of embeddings coloured by sentiment.

Text Preprocessing & Feature Engineering

clean_text:

Removes URLs, HTML tags, emojis, and special characters

Collapses whitespace and truncates to a configurable max length.

build_document:

Combines product_name, Summary, and Review into one rich document with pipe separators.

Fast‑mode stratified sampling by sentiment to ~20K documents for efficient experimentation.

Embedding Generation

Uses Mistral cloud embeddings if MISTRAL_API_KEY is set (mistral-embed, 1024‑dim), with validation that vectors are L2-normalised.

Falls back to TF‑IDF + TruncatedSVD (256‑dim) if Mistral is unavailable.

Embeddings and doc IDs cached to flipkart_embeddings.npy and flipkart_doc_ids.npy with shape checks.

FAISS Vector Index

FAISSIndex wrapper over IndexFlatIP:

Build, save, load, and search methods

Index persistence to flipkart.faiss

Self‑retrieval sanity check (top‑1 retrieval for first few docs).

Semantic Search Engine

SemanticSearchEngine exposes a high-level .search() API.

Returns a DataFrame with:

score (cosine similarity)

doc_idx (true corpus index, fixed in v2)

rank, product_name, Review, Rate, Sentiment, etc.

Supports:

Over‑fetching to compensate for post‑filtering

Optional sentiment filter (positive|negative|neutral)

Optional minimum rating filter.

RAG Pipeline

RAGPipeline wires the search engine and Mistral chat model together.

System prompt: “expert Flipkart product advisor” constrained to use only retrieved reviews and to answer honestly if evidence is insufficient.

Builds compact textual context from top‑K reviews with product name, rating, sentiment, similarity, and truncated review text.

Returns a structured response dict: query, answer, retrieved_docs, context_used.

Evaluation Suite

Implements classic ranking metrics:

precision_at_k, recall_at_k, f1_at_k

ndcg_at_k with binary relevance

mean_reciprocal_rank (MRR).

All metrics have unit tests to catch edge cases (empty relevance sets, K=0, etc.).

Synthetic evaluation set builder:

Samples products with ≥3 reviews

Uses product name prefixes as queries

Treats all reviews of that product as the relevant set.

Getting Started
Prerequisites
Python 3.x (Colab runtime or local environment) with:

mistralai

faiss-cpu

pandas, numpy

scikit-learn

matplotlib, seaborn

tqdm

wordcloud (optional for word clouds).

Flipkart reviews dataset ZIP (e.g. archive__1_.zip containing Dataset-SA.csv).

Mistral API key (optional but recommended for cloud embeddings and RAG answers).

Setup
Clone / open the notebook

Open Vector Search.ipynb in Google Colab or your local Jupyter environment.

Configure the dataset

Place the dataset ZIP in one of the auto‑detected locations, or set CFG.zip_path manually.

Set the Mistral API key (optional)

python
import os
os.environ["MISTRAL_API_KEY"] = "your-key-here"
If not set, the pipeline will use the local TF‑IDF + SVD embedding backend.

Run the notebook sections in order

Environment Setup & Configuration

Mistral Client (Cache + Retry + Fallback)

Data Loading & Validation

EDA & Visualisations

Text Preprocessing & Feature Engineering

Embedding Generation

FAISS Vector Index

Semantic Search Engine

RAG Pipeline

Evaluation.

Usage
Semantic Search
python
query = "affordable cooler for small room"
results = engine.search(query, k=3)
engine.pretty_print(results, query=query)
This returns the top‑3 most similar reviews, optionally filtered by sentiment or minimum rating.

RAG Question Answering
python
question = "Which air cooler is best for a small room under 5000 rupees?"
resp = rag.answer(question, k=5)
rag.print_answer(resp)
The pipeline retrieves the top‑K reviews, builds context, and asks the Mistral chat model to answer using only those reviews.

Evaluation
After building the synthetic evaluation set, you can compute metrics across multiple Ks (e.g. 1, 3, 5, 10) to validate retrieval quality.

Design Highlights & Best Practices
Clear separation of concerns: configuration, data loading, preprocessing, embedding, indexing, retrieval, generation, and evaluation live in distinct, readable blocks.

Defensive coding: extensive sanity checks (embedding norms, FAISS self‑retrieval, metric unit tests, config validation).

Reproducibility: fixed random seeds for sampling, PCA, and evaluation query selection.

Production‑minded design:

Persistent FAISS index and embedding caches

API caching to control latency and cost

Easy hooks for deployment (e.g. wrapping SemanticSearchEngine and RAGPipeline behind a REST API).
