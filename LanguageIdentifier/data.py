# ========================
# Multilingual Retrieval Chatbot
# English + Hindi + Hinglish → English Answers
# ========================

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
import re

print("Loading dataset...")
# Load the Flan v2 subset from AI4Bharat (Hindi + English instructions/QA)
ds = load_dataset("ai4bharat/indic-instruct-data-v0.1", "flan_v2")

# Extract questions (inputs) and answers (outputs)
questions = []
answers = []

factory = IndicNormalizerFactory()
normalizer_hi = factory.get_normalizer("hi")

print("Processing dataset...")
# Use both Hindi and English splits for better coverage
for split in ['en', 'hi']:
    print(f"Processing {split} split...")
    for example in ds[split]:
        # Use 'inputs' field (question/instruction)
        full_input = example.get('inputs', '').strip()
        # Use 'targets' field (answer/response)
        output = example.get('targets', '').strip()
        
        if full_input and output:
            # Normalize Hindi text (helps with variations)
            if any('\u0900' <= c <= '\u097f' for c in full_input):  # Contains Devanagari
                full_input = normalizer_hi.normalize(full_input)
            
            questions.append(full_input)
            answers.append(output)

print(f"Loaded {len(questions)} Q&A pairs")

# ========================
# Build Semantic Search Index
# ========================

print("Loading multilingual embedding model...")
# Best free model for English + Hindi + Hinglish
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Creating embeddings (this may take a minute)...")
embeddings = model.encode(questions, batch_size=32, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Normalize embeddings for better cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
index.add(embeddings)

print(f"Index built with {index.ntotal} entries")

# ========================
# Chatbot Loop
# ========================

def preprocess_query(query):
    # Basic normalization for Hindi
    if any('\u0900' <= c <= '\u097f' for c in query):
        query = normalizer_hi.normalize(query)
    return query.lower().strip()

def find_best_answer(user_query, top_k=1):
    processed = preprocess_query(user_query)
    query_emb = model.encode([processed])
    query_emb = np.array(query_emb).astype('float32')
    faiss.normalize_L2(query_emb)
    
    scores, indices = index.search(query_emb, top_k)
    best_idx = indices[0][0]
    confidence = scores[0][0]
    
    return answers[best_idx], confidence

print("\nChatbot is ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "bye", "बाय"]:
        print("Bot: Goodbye! 😊")
        break
    
    if not user_input:
        continue
    
    answer, score = find_best_answer(user_input)
    
    # Optional: Only reply if confidence is decent
    if score > 0.65:  # Tune this threshold as needed
        print("Bot:", answer)
    else:
        print("Bot: Sorry, I don't have a good answer for that yet. Try rephrasing!")
    
    print(f"(Confidence: {score:.3f})\n")