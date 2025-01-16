import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chains.summarize import load_summarize_chain

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Ensure the API key is configured properly
if not google_api_key:
    st.error("Google API key is not set. Please check your .env file.")
else:
    genai.configure(api_key=google_api_key)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
            else:
                st.warning("Some pages in the PDF could not be read.")
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and store vector embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get the conversational QA chain for the agent
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say,
    'Answer is not available in the context', and don't provide a wrong answer.

    Context:\n{context}\n
    Question: {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Tool: PDF retrieval with summarization
def pdf_tool_with_summary(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(query)
    
    # Summarize relevant documents first
    summarization_chain = load_summarize_chain(ChatGoogleGenerativeAI(model="gemini-pro"), chain_type="stuff")
    summary = summarization_chain({"input_documents": docs}, return_only_outputs=True)
    
    # Generate final response based on the summarized content
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    
    # If found in PDF, mention that the answer is from the PDF
    if response.get('output_text', 'Answer not found in PDF.') != 'Answer not found in PDF.':
        return f"Answer from PDF: {response.get('output_text', 'Answer not found in PDF.')}"
    else:
        return "Answer is not available in the context of the PDF."

# Tool: General Knowledge LLM-based reasoning
def general_knowledge_tool(query):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    response = model.invoke(query)
    
    # Correct way to get the response content from an AIMessage object
    return f"Answer is not from PDF, it's based on general knowledge: {response.content}"

# Create LangChain tools
pdf_tool_instance = Tool(
    name="PDF Retrieval Tool",
    func=pdf_tool_with_summary,
    description="Use this tool to answer questions based on the PDF documents."
)

general_knowledge_tool_instance = Tool(
    name="General Knowledge Tool",
    func=general_knowledge_tool,
    description="Use this tool to answer general knowledge questions outside the context of the PDF."
)

# Initialize the LangChain agent with both tools and the LLM
def initialize_langchain_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)  # Add the LLM here
    tools = [pdf_tool_instance, general_knowledge_tool_instance]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent

# Agent-based response function
def agent_respond(agent, query):
    response = agent.invoke([query])  # Use invoke() instead of run()
    return response

# Handle user input and query the agent
def user_input(user_question):
    agent = initialize_langchain_agent()
    response = agent_respond(agent, user_question)
    st.write("Reply: ", response)

# Streamlit main app function
def main():
    st.set_page_config("Chat with PDF", layout="wide")
    st.header("Chat with PDF 💬")

    # Provide testing options for users to ask questions
    user_question = st.text_input("Ask a Question")

    if user_question:
        response = user_input(user_question)
        st.write("Response: ", response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():  # Ensure the PDF has valid text
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Text processing and vector store creation completed.")
                    else:
                        st.error("Uploaded PDF contains no readable text.")
            else:
                st.error("Please upload at least one PDF file.")
        
if __name__ == "__main__":
    main()