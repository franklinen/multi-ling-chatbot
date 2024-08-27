# RAG-Based Legal Bot

## Overview

This project implements a Retrieval Augmented Generation (RAG) bot to answer questions about legal agreements. It leverages Large Language Models (LLMs) from the Huggingface platform and a vector database(FAISS) to provide contextually relevant responses. The model used allows for multilingual embeddings and so the bot can interact in multiple languages.

## Background and Motivation

Legal documents are often complex and difficult to navigate. This project aims to simplify the process of finding information within these documents using a question-answering approach. 

## Datasets

The project utilizes a set of PDF legal agreements as its knowledge base. These documents are preprocessed, split into chunks, and embedded into a vector database for efficient retrieval.

## Practical Applications

This RAG bot can be used by legal professionals, advisors, or anyone seeking to understand specific details within legal agreements. It can save time and effort by quickly providing targeted information.

## Milestones

- **Data Preprocessing:** Loading, splitting, and embedding legal documents.
- **Vector Database Creation:** Storing document embeddings for efficient retrieval.
- **Query Embedding and Search:** Generating embeddings for user queries and retrieving relevant document chunks.
- **Response Generation:** Leveraging LLMs to provide contextually relevant answers based on retrieved information.

## References

- LangChain: A framework for developing applications powered by language models.
- Hugging Face Transformers: A library for natural language processing tasks.
- FAISS: A library for efficient similarity search and clustering of dense vectors.