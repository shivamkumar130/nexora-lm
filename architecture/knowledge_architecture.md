# Nexora Knowledge Architecture

## Objective

The Nexora Knowledge System enables semantic retrieval of information from processed datasets.

## Pipeline

PDF Documents

↓

Text Extraction

↓

Text Cleaning

↓

Metadata Generation

↓

Chunk Generation

↓

Embedding Generation

↓

FAISS Vector Database

↓

Retriever Engine

↓

Semantic Search

## Components

### Embedding Generator

Converts text chunks into dense vector representations using BAAI/bge-small-en-v1.5.

### FAISS Database

Stores vector embeddings for fast similarity search.

### Retriever

Converts user queries into embeddings and retrieves the most relevant chunks.

### Semantic Search

Displays retrieved chunks and their content.

## Current Status

FAISS Index: Complete

Retriever Engine: Complete

Semantic Search: Complete

Version: 0.1
