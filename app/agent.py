import os
import logging
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .nasa_api import fetch_imagery
from .image_utils import detect_changes, preprocess_image
from .classifier import classify_image
from .prompts import SOLUTION_PROMPT, SUMMARY_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SatelliteAgentConfig:
    """Configuration class for Satellite Agent"""
    
    def __init__(
        self,
        knowledge_base_path: str = "knowledge_base/disaster_solutions.txt",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "llama3-8b-8192",
        llm_temperature: float = 0.7,
        summary_temperature: float = 0.4,
        confidence_threshold: float = 0.6,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        cache_dir: str = "./cache",
        max_retries: int = 3
    ):
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.summary_temperature = summary_temperature
        self.confidence_threshold = confidence_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        
        Path(cache_dir).mkdir(parents=True, exist_ok=True)


class RAGChainBuilder:
    """Builder class for RAG chain with caching and error handling"""
    
    def __init__(self, config: SatelliteAgentConfig):
        self.config = config
        self._vectorstore = None
        self._embeddings = None
        
    @lru_cache(maxsize=1)
    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        logger.info(f"Loading embeddings model: {self.config.embedding_model}")
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            cache_folder=self.config.cache_dir
        )
    
    def _load_and_split_documents(self) -> list:
        try:
            logger.info(f"Loading knowledge base from: {self.config.knowledge_base_path}")
            
            if not os.path.exists(self.config.knowledge_base_path):
                raise FileNotFoundError(
                    f"Knowledge base not found: {self.config.knowledge_base_path}"
                )
            
            loader = TextLoader(self.config.knowledge_base_path)
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            
            logger.info(f"Split knowledge base into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def _build_vectorstore(self) -> FAISS:
        try:
            chunks = self._load_and_split_documents()
            embeddings = self._get_embeddings()
            
            logger.info("Building FAISS vectorstore...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            cache_path = os.path.join(self.config.cache_dir, "vectorstore")
            vectorstore.save_local(cache_path)
            logger.info(f"Vectorstore cached at: {cache_path}")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error building vectorstore: {str(e)}")
            raise
    
    def get_vectorstore(self) -> FAISS:
        if self._vectorstore is not None:
            return self._vectorstore
        
        cache_path = os.path.join(self.config.cache_dir, "vectorstore")
        
        if os.path.exists(cache_path):
            try:
                logger.info("Loading vectorstore from cache...")
                embeddings = self._get_embeddings()
                self._vectorstore = FAISS.load_local(
                    cache_path, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vectorstore loaded from cache")
                return self._vectorstore
            except Exception as e:
                logger.warning(f"Failed to load cached vectorstore: {str(e)}")
        
        self._vectorstore = self._build_vectorstore()
        return self._vectorstore
    
    def build_rag_chain(self):
        try:
            vectorstore = self.get_vectorstore()
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            
            llm = ChatGroq(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_retries=self.config.max_retries
            )
            
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
            
            logger.info("RAG chain built successfully")
            return rag_chain
            
        except Exception as e:
            logger.error(f"Error building RAG chain: {str(e)}")
            raise


class ClassificationAgent:
    """Classification agent with summary generation"""
    
    def __init__(self, config: SatelliteAgentConfig):
        self.config = config
        self.summary_llm = ChatGroq(
            model=config.llm_model,
            temperature=config.summary_temperature,
            max_retries=config.max_retries
        )
        
    def is_uncertain(self, output: Dict[str, Any]) -> bool:
        return output.get("confidence", 0) < self.config.confidence_threshold
    
    def build_chain(self):
        image_preprocessor = RunnableLambda(preprocess_image)
        classifier_chain = RunnableLambda(classify_image)
        
        summary_chain = (
            RunnableLambda(lambda result: {"label": result["label"]})
            | SUMMARY_PROMPT
            | self.summary_llm
            | StrOutputParser()
        )
        
        def process_classification(result: Dict[str, Any]) -> Dict[str, Any]:
            try:
                summary = (
                    summary_chain.invoke(result)
                    if not self.is_uncertain(result)
                    else "⚠️ Model uncertainty detected — confidence below threshold. Human review recommended."
                )
                
                return {
                    "classification": result,
                    "summary": summary,
                }
            except Exception as e:
                logger.error(f"Error in classification processing: {str(e)}")
                return {
                    "classification": result,
                    "summary": f"Error generating summary: {str(e)}",
                }
        
        classification_agent = (
            image_preprocessor
            | classifier_chain
            | RunnableLambda(process_classification)
        )
        
        return classification_agent


class SatelliteAgent:
    """Main Satellite Disaster Analysis Agent"""
    
    def __init__(self, config: Optional[SatelliteAgentConfig] = None):
        self.config = config or SatelliteAgentConfig()
        
        logger.info("Initializing Satellite Agent...")
        self.rag_builder = RAGChainBuilder(self.config)
        self.rag_chain = self.rag_builder.build_rag_chain()
        self.classification_agent = ClassificationAgent(self.config).build_chain()
        logger.info("Satellite Agent initialized successfully")
    
    def fetch_satellite_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Fetching satellite data for location: {input_data.get('location')}")
            
            required_fields = ["location", "before_date", "after_date"]
            for field in required_fields:
                if field not in input_data:
                    raise ValueError(f"Missing required field: {field}")
            
            before_img = fetch_imagery(
                input_data["location"], 
                input_data["before_date"]
            )
            after_img = fetch_imagery(
                input_data["location"], 
                input_data["after_date"]
            )
            
            logger.info("Detecting changes between images...")
            diff_map = detect_changes(before_img, after_img)
            
            return {
                "images": {
                    "before": before_img,
                    "after": after_img,
                    "difference": diff_map
                },
                "input_params": input_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching satellite data: {str(e)}")
            raise
    
    def run_classification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Running classification analysis...")
            diff_map = data["images"]["difference"]
            analysis_results = self.classification_agent.invoke(diff_map)
            
            logger.info(
                f"Classification complete: {analysis_results['classification']['label']} "
                f"(confidence: {analysis_results['classification'].get('confidence', 'N/A')})"
            )
            
            return {**data, **analysis_results}
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return {
                **data,
                "classification": {"label": "Error", "confidence": 0.0},
                "summary": f"Classification error: {str(e)}"
            }
    
    def generate_solutions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            label = data["classification"]["label"]
            
            if label in ["No Significant Change", "Error"]:
                logger.info("No significant change detected or error occurred")
                data["solutions"] = (
                    "✓ No significant change detected. No immediate action required."
                    if label == "No Significant Change"
                    else "⚠️ Unable to generate solutions due to classification error."
                )
            else:
                logger.info(f"Generating solutions for: {label}")
                data["solutions"] = self.rag_chain.invoke(data)
                logger.info("Solutions generated successfully")
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating solutions: {str(e)}")
            data["solutions"] = f"Error generating solutions: {str(e)}"
            return data
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("=" * 50)
            logger.info("Starting satellite analysis pipeline")
            logger.info("=" * 50)
            
            data = self.fetch_satellite_data(input_data)
            data = self.run_classification(data)
            data = self.generate_solutions(data)
            
            logger.info("=" * 50)
            logger.info("Analysis pipeline completed successfully")
            logger.info("=" * 50)
            
            return data
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            raise
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.analyze(input_data)


def create_satellite_agent(config: Optional[SatelliteAgentConfig] = None) -> SatelliteAgent:
    return SatelliteAgent(config)


if __name__ == "__main__":
    config = SatelliteAgentConfig(
        knowledge_base_path="knowledge_base/disaster_solutions.txt",
        confidence_threshold=0.65,
        llm_temperature=0.7
    )
    
    agent = create_satellite_agent(config)
    
    input_data = {
        "location": "34.0522,-118.2437",
        "before_date": "2024-01-01",
        "after_date": "2024-01-15"
    }
    
    try:
        results = agent.analyze(input_data)
        
        print("\n" + "=" * 50)
        print("ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Classification: {results['classification']['label']}")
        print(f"Confidence: {results['classification'].get('confidence', 'N/A')}")
        print(f"\nSummary:\n{results['summary']}")
        print(f"\nRecommended Solutions:\n{results['solutions']}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")