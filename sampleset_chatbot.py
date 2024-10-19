# -*- coding: utf-8 -*-

# Part 1: Building the RAG Model

# Import necessary libraries
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from cohere import Client
import pdfplumber
import streamlit as st
import cohere

# Prompt user to upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Ensure there's an uploaded file before proceeding
if uploaded_file is not None:
    # Use pdfplumber to extract text from the uploaded PDF
    def extract_text_from_pdf(uploaded_file):
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"  # Extract text from each page
        return text

    # Extract and display the text from the PDF
    document_text = extract_text_from_pdf(uploaded_file)

# Generate Document Embeddings
# create embeddings from the text extracted from the PDF
from sentence_transformers import SentenceTransformer

# Load pre-trained transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate sentence-level embeddings for the text
sentences = document_text.split(".")  # Split text into sentences
sentence_embeddings = model.encode(sentences)

# Example usage: print one sentence and its corresponding embedding
print(sentences[0])
print(sentence_embeddings[0])

# Set your Pinecone API key (ensure to replace with your actual API key)
api_key = '3df05ff6-0236-4da1-b8f5-f803a6d00eb1'

# Initialize the Pinecone instance
pc = Pinecone(api_key=api_key)

# Define the index name and embedding dimension
index_name = 'document-embeddings'
embedding_dim = sentence_embeddings.shape[1]  # Ensure sentence_embeddings is already defined

# Check if the index already exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dim,
        metric='euclidean',  # You can adjust the metric based on your needs
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # You can specify your region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Upload the embeddings to the index
for i, embedding in enumerate(sentence_embeddings):
    index.upsert([(f'sentence-{i}', embedding)])

print("Embeddings successfully uploaded to Pinecone!")

import numpy as np
from sentence_transformers import util

# Function to find the most similar question
def get_most_relevant_sentence(query, model, sentences, sentence_embeddings):
    # Generate embedding for the query
    query_embedding = model.encode(query)

    # Calculate cosine similarity with each sentence embedding
    similarities = util.cos_sim(query_embedding, sentence_embeddings)

    # Get the index of the most similar sentence
    best_idx = np.argmax(similarities)

    return sentences[best_idx]

# Example usage
user_query = "What are pathogens"
most_relevant_sentence = get_most_relevant_sentence(user_query, model, sentences, sentence_embeddings)
print(f"Most relevant sentence: {most_relevant_sentence}")

# Cohere API for Generating a Response
# Initialize the Cohere client
co = cohere.Client('eiVpA3k8j4oQXtOqRsHGWaW3tLHJqcaoMlkdGfG8')

def generate_answer(context, query):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    # Generate a response using Cohere
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=100
    )
    return response.generations[0].text

# Example usage
context = most_relevant_sentence
answer = generate_answer(context, user_query)
print(f"Generated answer: {answer}")

# Part 2: Interactive QA Bot Interface
# Build Frontend Using Streamlit
# Create an interface where users can upload PDFs and ask questions
# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone_client = pinecone.Pinecone(api_key='3df05ff6-0236-4da1-b8f5-f803a6d00eb1')  # Replace with your Pinecone API key
index = pinecone_client.Index('document-embeddings')

# Initialize Cohere
co = cohere.Client('eiVpA3k8j4oQXtOqRsHGWaW3tLHJqcaoMlkdGfG8')  # Replace with your Cohere API key

# Extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Streamlit App
st.title("QA Bot: Ask Questions from Your PDF")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    document_text = extract_text_from_pdf(uploaded_file)
    st.write("Document uploaded successfully!")

    # Process document for embeddings
    sentences = document_text.split(".")
    sentence_embeddings = model.encode(sentences)

    # Handle user queries
    user_query = st.text_input("Ask a question:")

    if user_query:
        # Find the most relevant sentence
        relevant_sentence = get_most_relevant_sentence(user_query, model, sentences, sentence_embeddings)
        st.write(f"Relevant sentence: {relevant_sentence}")

        # Generate a full answer using Cohere
        answer = generate_answer(relevant_sentence, user_query)
        st.write(f"Answer: {answer}")

# Add multiple query handling
user_query = st.text_input("Ask a question:")

if user_query:
    query_embedding = model.encode(user_query)
    relevant_docs = retrieve_relevant_docs(query_embedding)
    context = ' '.join([doc['metadata']['text'] for doc in relevant_docs['matches']])

    # Generate answer using Cohere
    answer = generate_answer(context, user_query)
    st.write(f"Answer: {answer}")

    # Show relevant document segments
    st.write("Relevant document segments:")
    for doc in relevant_docs['matches']:
        st.write(doc['metadata']['text'])
