#!/usr/bin/env python3
"""
Comprehensive test suite for Intelligent Knowledge Base
Run with: python test_prototype.py
"""

import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add the lib directory to the path
sys.path.append(str(Path(__file__).parent / "lib"))

class TestKnowledgeBase(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.test_docs_dir = Path("test_documents")
        self.test_docs_dir.mkdir(exist_ok=True)
        
        # Create test documents
        self.create_test_documents()
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove test documents
        for file in self.test_docs_dir.glob("*"):
            file.unlink()
        self.test_docs_dir.rmdir()
        
        # Remove chroma_db if exists
        chroma_path = Path("./chroma_db")
        if chroma_path.exists():
            for file in chroma_path.glob("*"):
                file.unlink()
            chroma_path.rmdir()
    
    def create_test_documents(self):
        """Create test documents for testing"""
        
        # Test PDF content
        pdf_content = """
        PROJECT OVERVIEW
        ================
        
        Project Name: AI Knowledge Base Implementation
        Project Manager: Jane Smith
        Start Date: 2024-01-15
        End Date: 2024-03-30
        
        KEY OBJECTIVES:
        1. Develop intelligent document search capability
        2. Implement AI-powered Q&A system
        3. Ensure 90%+ answer accuracy
        4. Provide source citations for all answers
        
        DELIVERABLES:
        - Working prototype by Week 8
        - Production deployment by Week 12
        - User documentation
        - Technical specifications
        
        RISKS:
        - API rate limits may affect performance
        - Large documents may require chunking optimization
        - Data privacy concerns with external APIs
        """
        
        # Create a simple text file
        text_file = self.test_docs_dir / "project_plan.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(pdf_content)
        
        # Create a requirements file
        requirements_content = """
        DEPENDENCIES:
        - Python 3.8+
        - Streamlit for web interface
        - ChromaDB for vector storage
        - OpenAI GPT for answer generation
        - Sentence transformers for embeddings
        
        SETUP INSTRUCTIONS:
        1. Install requirements: pip install -r requirements.txt
        2. Set OPENAI_API_KEY environment variable
        3. Run: streamlit run app.py
        
        TESTING:
        - Unit tests: python test_prototype.py
        - Integration: Manual testing with sample documents
        - Performance: Test with 50+ page documents
        """
        
        requirements_file = self.test_docs_dir / "requirements.txt"
        with open(requirements_file, 'w', encoding='utf-8') as f:
            f.write(requirements_content)
    
    def test_document_processing(self):
        """Test document processing functionality"""
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test with text file
        test_file = self.test_docs_dir / "project_plan.txt"
        with open(test_file, 'rb') as f:
            class MockUploadedFile:
                name = "project_plan.txt"
                size = os.path.getsize(test_file)
                
                def getvalue(self):
                    with open(test_file, 'rb') as file:
                        return file.read()
            
            mock_files = [MockUploadedFile()]
            
            # Process documents
            collection = processor.process_documents(mock_files)
            
            # Verify collection was created
            self.assertIsNotNone(collection)
            
            # Check if documents were added
            results = collection.get()
            self.assertGreater(len(results['ids']), 0)
    
    def test_qa_engine(self):
        """Test Q&A engine functionality"""
        from qa_engine import QAEngine
        
        engine = QAEngine()
        
        # Mock a collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'documents': [['Test document content about project objectives and deliverables.']],
            'metadatas': [[{'source': 'test.pdf', 'similarity_score': 0.85}]],
            'distances': [[0.15]]
        }
        
        # Test question answering
        question = "What are the project objectives?"
        chunks, sources = engine.find_relevant_chunks(question, mock_collection)
        
        self.assertGreater(len(chunks), 0)
        self.assertGreater(len(sources), 0)
    
    def test_file_validation(self):
        """Test file validation logic"""
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test valid file
        valid_file = self.test_docs_dir / "project_plan.txt"
        self.assertTrue(processor.validate_file(str(valid_file), os.path.getsize(valid_file)))
        
        # Test invalid file (too large)
        with self.assertRaises(Exception):
            processor.validate_file("test.pdf", 100 * 1024 * 1024)  # 100MB
    
    def test_smart_chunking(self):
        """Test text chunking functionality"""
        from document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        test_text = """
        This is a test document. It contains multiple sentences. 
        Each sentence should be properly chunked. The chunking algorithm 
        should preserve context and create reasonable sized chunks.
        
        This is a new paragraph. It should be treated separately from 
        the previous paragraph for better context preservation.
        """
        
        chunks = processor.smart_chunk_text(test_text, "test.txt")
        
        self.assertGreater(len(chunks), 0)
        
        # Check chunk structure
        for chunk, metadata in chunks:
            self.assertIsInstance(chunk, str)
            self.assertIsInstance(metadata, dict)
            self.assertIn('source', metadata)

def run_performance_test():
    """Run performance tests with timing"""
    import time
    from document_processor import DocumentProcessor
    from qa_engine import QAEngine
    
    print("\n" + "="*50)
    print("PERFORMANCE TESTING")
    print("="*50)
    
    # Create a larger test document
    large_content = "Test content. " * 1000  # ~15KB
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(large_content)
        temp_path = f.name
    
    try:
        # Test processing time
        processor = DocumentProcessor()
        
        start_time = time.time()
        
        class MockUploadedFile:
            name = "large_test.txt"
            
            def getvalue(self):
                with open(temp_path, 'rb') as file:
                    return file.read()
        
        mock_files = [MockUploadedFile()]
        collection = processor.process_documents(mock_files)
        
        processing_time = time.time() - start_time
        print(f"📊 Document processing time: {processing_time:.2f}s")
        
        # Test query time
        engine = QAEngine()
        
        start_time = time.time()
        chunks, sources = engine.find_relevant_chunks("test content", collection)
        query_time = time.time() - start_time
        
        print(f"📊 Query processing time: {query_time:.2f}s")
        
        # Performance thresholds (adjust based on your needs)
        assert processing_time < 30.0, "Processing too slow"
        assert query_time < 5.0, "Query too slow"
        
        print("✅ Performance tests passed!")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(exit=False)
    
    # Run performance tests
    run_performance_test()
    
    print("\n" + "="*50)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*50)