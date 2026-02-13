import pytest
from unittest.mock import Mock, patch, MagicMock
from app.agent import (
    SatelliteAgentConfig,
    RAGChainBuilder,
    ClassificationAgent,
    SatelliteAgent,
    create_satellite_agent
)


class TestSatelliteAgentConfig:
    """Test SatelliteAgentConfig configuration"""
    
    def test_config_default_values(self):
        """Test default configuration values"""
        config = SatelliteAgentConfig()
        assert config.llm_model == "llama3-8b-8192"
        assert config.llm_temperature == 0.7
        assert config.confidence_threshold == 0.6
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
    
    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = SatelliteAgentConfig(
            llm_temperature=0.5,
            confidence_threshold=0.8,
            chunk_size=1000
        )
        assert config.llm_temperature == 0.5
        assert config.confidence_threshold == 0.8
        assert config.chunk_size == 1000
    
    def test_config_cache_dir_creation(self, test_cache_dir):
        """Test cache directory is created"""
        config = SatelliteAgentConfig(cache_dir=test_cache_dir)
        assert config.cache_dir == test_cache_dir


class TestRAGChainBuilder:
    """Test RAG chain builder"""
    
    @pytest.fixture
    def config(self, temp_knowledge_base, test_cache_dir):
        """Create test configuration"""
        return SatelliteAgentConfig(
            knowledge_base_path=temp_knowledge_base,
            cache_dir=test_cache_dir
        )
    
    @pytest.fixture
    def rag_builder(self, config):
        """Create RAG chain builder"""
        return RAGChainBuilder(config)
    
    def test_rag_builder_initialization(self, rag_builder):
        """Test RAG builder initializes"""
        assert rag_builder is not None
        assert rag_builder.config is not None
    
    @patch('app.agent.HuggingFaceEmbeddings')
    def test_get_embeddings(self, mock_embeddings, rag_builder):
        """Test embeddings loading"""
        mock_embeddings.return_value = Mock()
        embeddings = rag_builder._get_embeddings()
        assert embeddings is not None
    
    def test_load_and_split_documents(self, rag_builder):
        """Test document loading and splitting"""
        chunks = rag_builder._load_and_split_documents()
        assert chunks is not None
        assert len(chunks) > 0
    
    def test_load_missing_knowledge_base(self, test_cache_dir):
        """Test error when knowledge base is missing"""
        config = SatelliteAgentConfig(
            knowledge_base_path="nonexistent.txt",
            cache_dir=test_cache_dir
        )
        builder = RAGChainBuilder(config)
        with pytest.raises(FileNotFoundError):
            builder._load_and_split_documents()


class TestClassificationAgent:
    """Test classification agent"""
    
    @pytest.fixture
    def config(self):
        """Create test config"""
        return SatelliteAgentConfig()
    
    @pytest.fixture
    @patch('app.agent.ChatGroq')
    def classification_agent(self, mock_groq, config):
        """Create classification agent with mocked LLM"""
        mock_groq.return_value = Mock()
        return ClassificationAgent(config)
    
    def test_classification_agent_initialization(self, classification_agent):
        """Test agent initializes"""
        assert classification_agent is not None
        assert classification_agent.config is not None
    
    def test_is_uncertain_below_threshold(self, classification_agent):
        """Test uncertainty detection below threshold"""
        result = {"confidence": 0.5}
        assert classification_agent.is_uncertain(result) is True
    
    def test_is_uncertain_above_threshold(self, classification_agent):
        """Test uncertainty detection above threshold"""
        result = {"confidence": 0.9}
        assert classification_agent.is_uncertain(result) is False
    
    def test_is_uncertain_at_threshold(self, classification_agent):
        """Test uncertainty at exact threshold"""
        result = {"confidence": 0.6}
        assert classification_agent.is_uncertain(result) is False


class TestSatelliteAgent:
    """Test main satellite agent"""
    
    @pytest.fixture
    @patch('app.agent.RAGChainBuilder')
    @patch('app.agent.ClassificationAgent')
    def agent(self, mock_class_agent, mock_rag_builder):
        """Create agent with mocked components"""
        mock_rag_builder.return_value.build_rag_chain.return_value = Mock()
        mock_class_agent.return_value.build_chain.return_value = Mock()
        return SatelliteAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes"""
        assert agent is not None
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'rag_chain')
        assert hasattr(agent, 'classification_agent')
    
    @patch('app.agent.fetch_imagery')
    @patch('app.agent.detect_changes')
    def test_fetch_satellite_data(self, mock_detect, mock_fetch, agent, mock_analysis_input):
        """Test satellite data fetching"""
        mock_fetch.return_value = Mock()
        mock_detect.return_value = Mock()
        
        result = agent.fetch_satellite_data(mock_analysis_input)
        
        assert result is not None
        assert 'images' in result
        assert 'input_params' in result
        assert 'timestamp' in result
    
    def test_fetch_satellite_data_missing_fields(self, agent):
        """Test error when required fields are missing"""
        with pytest.raises(ValueError):
            agent.fetch_satellite_data({"location": (0, 0)})
    
    @patch('app.agent.fetch_imagery')
    @patch('app.agent.detect_changes')
    def test_run_classification(self, mock_detect, mock_fetch, agent, mock_analysis_input):
        """Test classification execution"""
        mock_fetch.return_value = Mock()
        mock_detect.return_value = Mock()
        
        data = agent.fetch_satellite_data(mock_analysis_input)
        agent.classification_agent = Mock()
        agent.classification_agent.invoke.return_value = {
            "classification": {"label": "wildfire", "confidence": 0.9},
            "summary": "Test summary"
        }
        
        result = agent.run_classification(data)
        
        assert result is not None
        assert 'classification' in result
        assert 'summary' in result
    
    def test_generate_solutions_normal_case(self, agent):
        """Test solution generation for normal changes"""
        data = {
            "classification": {"label": "No Significant Change", "confidence": 0.95},
            "summary": "No changes detected"
        }
        
        result = agent.generate_solutions(data)
        
        assert result is not None
        assert 'solutions' in result
        assert "No significant change" in result['solutions'] or "no immediate action" in result['solutions'].lower()
    
    def test_generate_solutions_error_case(self, agent):
        """Test solution generation for errors"""
        data = {
            "classification": {"label": "Error", "confidence": 0.0},
            "summary": "Error occurred"
        }
        
        result = agent.generate_solutions(data)
        
        assert result is not None
        assert 'solutions' in result
    
    @patch('app.agent.fetch_imagery')
    @patch('app.agent.detect_changes')
    def test_invoke_method(self, mock_detect, mock_fetch, agent, mock_analysis_input):
        """Test invoke method calls analyze"""
        mock_fetch.return_value = Mock()
        mock_detect.return_value = Mock()
        agent.classification_agent = Mock()
        agent.classification_agent.invoke.return_value = {
            "classification": {"label": "wildfire", "confidence": 0.9},
            "summary": "Test"
        }
        agent.rag_chain = Mock()
        agent.rag_chain.invoke.return_value = "Test solutions"
        
        result = agent.invoke(mock_analysis_input)
        
        assert result is not None


class TestCreateSatelliteAgent:
    """Test agent factory function"""
    
    @patch('app.agent.SatelliteAgent')
    def test_create_satellite_agent_default_config(self, mock_agent):
        """Test creating agent with default config"""
        mock_agent.return_value = Mock()
        agent = create_satellite_agent()
        assert agent is not None
        mock_agent.assert_called_once()
    
    @patch('app.agent.SatelliteAgent')
    def test_create_satellite_agent_custom_config(self, mock_agent):
        """Test creating agent with custom config"""
        mock_agent.return_value = Mock()
        config = SatelliteAgentConfig(llm_temperature=0.5)
        agent = create_satellite_agent(config)
        assert agent is not None
        mock_agent.assert_called_once_with(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
