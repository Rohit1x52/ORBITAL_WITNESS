import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

from .nasa_api import fetch_imagery
from .image_utils import detect_changes
from .classifier import classify_event
from .prompts import SOLUTION_PROMPT

# --- RAG Pipeline Setup ---

def setup_rag_chain():
    """Builds the RAG chain for solution generation."""
    # 1. Load Knowledge Base
    loader = TextLoader("knowledge_base/disaster_solutions.txt")
    documents = loader.load()

    # 2. Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 3. Create embeddings and vector store
    # Using a local, open-source model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # 4. Define the RAG chain
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)

    rag_chain = (
        {"context": retriever, "event_class": RunnablePassthrough(), "summary": RunnablePassthrough()}
        | SOLUTION_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Full Agentic Workflow ---

def create_satellite_agent():
    """
    Creates the main LCEL pipeline for satellite image analysis.
    This function assembles the entire agent workflow.
    """
    # Initialize the RAG chain once
    rag_chain = setup_rag_chain()

    # Define a function to handle the initial data fetching and primary analysis
    def initial_analysis(input_data: dict):
        """Fetches images, detects changes, and classifies the event."""
        before_img = fetch_imagery(input_data["location"], input_data["before_date"])
        after_img = fetch_imagery(input_data["location"], input_data["after_date"])
        diff_map = detect_changes(before_img, after_img)
        
        classification = classify_event(before_img, after_img, diff_map)
        
        return {
            "images": {"before": before_img, "after": after_img, "difference": diff_map},
            "analysis": classification,
            "input_params": input_data
        }

    # Define a function to route to the RAG chain
    def generate_solutions(analysis_output: dict):
        """
        Takes the classification result and invokes the RAG chain to get solutions.
        """
        analysis_data = analysis_output["analysis"]
        # The RAG chain is invoked with a dictionary matching its expected inputs
        solutions_text = rag_chain.invoke({
            "event_class": analysis_data["event_class"],
            "summary": analysis_data["summary"]
        })
        
        # Combine the initial analysis with the generated solutions
        analysis_output["solutions"] = solutions_text
        return analysis_output

    # Assemble the final agent using LCEL
    satellite_agent_chain = (
        RunnableLambda(initial_analysis)
        | RunnableLambda(generate_solutions)
    )

    return satellite_agent_chain