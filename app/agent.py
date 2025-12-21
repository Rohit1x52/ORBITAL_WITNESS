import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from .nasa_api import fetch_imagery
from .image_utils import detect_changes, preprocess_image
from .classifier import classify_image
from .prompts import SOLUTION_PROMPT, SUMMARY_PROMPT

def setup_rag_chain():
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

def create_satellite_agent():
    rag_chain = setup_rag_chain()
    image_preprocessor = RunnableLambda(preprocess_image)
    classifier_chain = RunnableLambda(classify_image)

    def is_uncertain(output):
        return output["confidence"] < 0.6
    summary_llm = ChatGroq(model="llama3-8b-8192", temperature=0.4)
    summary_chain = (
        RunnableLambda(lambda result: {"label": result["label"]})
        | SUMMARY_PROMPT
        | summary_llm
        | StrOutputParser()
    )
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
    def initial_data_fetch(input_data: dict):
        before_img = fetch_imagery(input_data["location"], input_data["before_date"])
        after_img = fetch_imagery(input_data["location"], input_data["after_date"])
        diff_map = detect_changes(before_img, after_img)
        
        return {
            "images": {"before": before_img, "after": after_img, "difference": diff_map},
            "input_params": input_data
        }

    def run_classification(data: dict):
        diff_map = data["images"]["difference"]
        analysis_results = classification_agent.invoke(diff_map)
        return {**data, **analysis_results}

    def generate_solutions(data: dict):
        if data["classification"]["label"] == "No Significant Change":
            data["solutions"] = "No significant change detected. No action required."
        else:
            data["solutions"] = rag_chain.invoke(data)
        return data
    satellite_agent_chain = (
        RunnableLambda(initial_data_fetch)
        | RunnableLambda(run_classification)
        | RunnableLambda(generate_solutions)
    )

    return satellite_agent_chain
