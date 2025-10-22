import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# --- Imports from our project ---
from .nasa_api import fetch_imagery
from .image_utils import detect_changes, preprocess_image
from .classifier import classify_image
from .prompts import SOLUTION_PROMPT, SUMMARY_PROMPT

# --- RAG Pipeline Setup (Same as before) ---
def setup_rag_chain():
    """Builds the RAG chain for solution generation."""
    loader = TextLoader("knowledge_base/disaster_solutions.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)

    rag_chain = (
        {
            "context": retriever,
            "event_class": lambda x: x["classification"]["label"],
            "summary": lambda x: x["summary"]
        }
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
    # Initialize the RAG chain
    rag_chain = setup_rag_chain()

    # --- This is YOUR LCEL logic for classification ---
    # Step 1: Preprocess image (resize etc.)
    image_preprocessor = RunnableLambda(preprocess_image)
    
    # Step 2: Classify satellite image
    classifier_chain = RunnableLambda(classify_image)
    
    # Step 3: Check if classification confidence is low
    def is_uncertain(output):
        return output["confidence"] < 0.6
    
    # Step 4: LLM summary generator
    summary_llm = ChatGroq(model="llama3-8b-8192", temperature=0.4)
    summary_chain = (
        RunnableLambda(lambda result: {"label": result["label"]})
        | SUMMARY_PROMPT
        | summary_llm
        | StrOutputParser()
    )
    
    # Step 5: Main Classification & Summary Pipeline
    # This chain will run on the 'diff_map' image
    classification_agent = (
        image_preprocessor
        | classifier_chain
        | RunnableLambda(
            lambda result: {
                "classification": result,
                "summary": (
                    summary_chain.invoke(result)
                    if not is_uncertain(result)
                    else "Model uncertain — human review recommended."
                ),
            }
        )
    )
    # --- End of your logic ---


    # --- These functions define the overall workflow ---
    def initial_data_fetch(input_data: dict):
        """Fetches images and detects changes."""
        before_img = fetch_imagery(input_data["location"], input_data["before_date"])
        after_img = fetch_imagery(input_data["location"], input_data["after_date"])
        diff_map = detect_changes(before_img, after_img)
        
        return {
            "images": {"before": before_img, "after": after_img, "difference": diff_map},
            "input_params": input_data
        }

    def run_classification(data: dict):
        """Runs your classification agent on the difference map."""
        diff_map = data["images"]["difference"]
        analysis_results = classification_agent.invoke(diff_map)
        
        # Combine the image data with the analysis results
        return {**data, **analysis_results}

    def generate_solutions(data: dict):
        """Runs the RAG chain to get solutions."""
        if data["classification"]["label"] == "No Significant Change":
            data["solutions"] = "No significant change detected. No action required."
        else:
            data["solutions"] = rag_chain.invoke(data)
        return data


    # --- Assemble the final agent ---
    satellite_agent_chain = (
        RunnableLambda(initial_data_fetch)
        | RunnableLambda(run_classification)
        | RunnableLambda(generate_solutions)
    )

    return satellite_agent_chain
