Project Overview:

This project is designed to build an Intelligent Q&A System that can answer queries based on PDF document content and general knowledge. It uses LangChain to orchestrate tool usage, leveraging a combination of document retrieval and large language models (LLMs). The system integrates a decision-making process to determine whether a query should be answered using document retrieval or general knowledge, ensuring the best possible response for each question.


Features:

PDF Document Retrieval: Allows the system to parse, index, and query PDFs to extract relevant information.
General Knowledge LLM Integration: Uses a pre-trained LLM to answer questions that do not require specific document context.
Agent-Based Tool Selection: An agent dynamically chooses whether to use document retrieval or general knowledge for answering a query based on the context.
FAISS for Efficient Search: FAISS indexing is used to enable fast and accurate similarity searches for chunked document embeddings.
Streamlit Frontend: A user-friendly interface built using Streamlit to input queries, view responses, and track tool usage.


Configuration:

Agents and Tool Decision Logic
The core of the system is built around an agent that selects the most appropriate tool based on the nature of the query:
Document Retrieval: Chosen when the query references specific content that is best answered with document context (e.g., "What are the main ideas from Chapter 3?").
LLM General Knowledge: Chosen for broad, well-known questions that do not need detailed document analysis (e.g., "What is space technology?").


Tools Used:

LangChain: For managing chains and agents.
SentenceTransformers: To generate embeddings for document chunks.
FAISS: For fast similarity searches through the embeddings.
Streamlit: Provides the frontend interface for users to interact with the system.
