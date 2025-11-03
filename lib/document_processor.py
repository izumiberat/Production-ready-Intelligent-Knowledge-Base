import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import re
from datetime import datetime
import traceback

class DocumentProcessor:
    def __init__(self):
        try:
            # Initialize with error handling
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Create or get collection with proper configuration
            self.collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise Exception(f"Failed to initialize document processor: {str(e)}")
    
    def validate_file(self, file_path: str, file_size: int) -> bool:
        """Validate file before processing"""
        # Check file size (max 50MB)
        if file_size > 50 * 1024 * 1024:
            raise Exception(f"File too large: {file_size // (1024*1024)}MB. Maximum size is 50MB.")
        
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            raise Exception("Temporary file not found")
        
        return True
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Enhanced PDF text extraction with comprehensive error handling"""
        try:
            self.validate_file(file_path, os.path.getsize(file_path))
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Validate PDF structure
                if len(reader.pages) == 0:
                    raise Exception("PDF appears to be empty or corrupted")
                
                if len(reader.pages) > 1000:
                    raise Exception(f"PDF too large: {len(reader.pages)} pages. Maximum is 1000 pages.")
                
                text = ""
                successful_pages = 0
                
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            # Clean up extracted text
                            page_text = re.sub(r'\s+', ' ', page_text).strip()
                            text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                            successful_pages += 1
                            
                    except Exception as e:
                        # Continue with other pages if one fails
                        print(f"Warning: Could not read page {page_num + 1}: {str(e)}")
                        continue
                
                if successful_pages == 0:
                    raise Exception("No text could be extracted from any page of the PDF")
                
                if successful_pages < len(reader.pages):
                    print(f"Warning: Extracted text from {successful_pages}/{len(reader.pages)} pages")
                
                return text
                
        except PyPDF2.PdfReadError as e:
            raise Exception(f"PDF reading error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Enhanced TXT file reading with comprehensive encoding detection"""
        self.validate_file(file_path, os.path.getsize(file_path))
        
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                    if text.strip():
                        return text
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                continue
        
        raise Exception("Could not read text file with common encodings. File may be binary or corrupted.")
    
    def smart_chunk_text(self, text: str, filename: str) -> List[Tuple[str, Dict]]:
        """Improved chunking that respects document structure with size validation"""
        if not text or len(text.strip()) < 10:
            raise Exception(f"Text too short or empty in file {filename}")
        
        chunks = []
        
        try:
            # Split by sections/headers first
            sections = re.split(r'\n-{3,}\s*Page \d+ -{3,}\n', text)
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Further split by sentences or natural breaks
                sentences = re.split(r'(?<=[.!?])\s+', section)
                
                current_chunk = ""
                current_length = 0
                target_chunk_size = 800
                overlap_size = 100
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    sentence_length = len(sentence.split())
                    
                    # Validate sentence isn't excessively long
                    if sentence_length > 500:
                        # Split very long sentences
                        words = sentence.split()
                        half_point = len(words) // 2
                        sentence = ' '.join(words[:half_point])
                        sentences.insert(sentences.index(sentence) + 1, ' '.join(words[half_point:]))
                        sentence_length = len(sentence.split())
                    
                    # If adding this sentence would exceed chunk size and we have content
                    if current_length + sentence_length > target_chunk_size and current_chunk:
                        chunks.append((current_chunk, {
                            "source": filename,
                            "chunk_type": "text",
                            "word_count": current_length,
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                        # Keep overlap for context
                        overlap_words = current_chunk.split()[-overlap_size:]
                        current_chunk = ' '.join(overlap_words) + " " + sentence
                        current_length = len(overlap_words) + sentence_length
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                            current_length += sentence_length
                        else:
                            current_chunk = sentence
                            current_length = sentence_length
                
                # Don't forget the last chunk of the section
                if current_chunk and current_length > 10:
                    chunks.append((current_chunk, {
                        "source": filename,
                        "chunk_type": "text",
                        "word_count": current_length,
                        "timestamp": datetime.now().isoformat()
                    }))
            
            # Validate we have chunks
            if not chunks:
                raise Exception(f"No valid chunks created from {filename}")
                
            return chunks
            
        except Exception as e:
            raise Exception(f"Error during text chunking for {filename}: {str(e)}")
    
    def process_documents(self, uploaded_files) -> chromadb.Collection:
        """Enhanced document processing with comprehensive error handling and progress tracking"""
        if not uploaded_files:
            raise Exception("No files provided for processing")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        all_embeddings = []
        
        total_files = len(uploaded_files)
        processed_files = 0
        
        for file_idx, uploaded_file in enumerate(uploaded_files, 1):
            print(f"Processing file {file_idx}/{total_files}: {uploaded_file.name}")
            
            # Validate file type
            if not (uploaded_file.name.lower().endswith('.pdf') or uploaded_file.name.lower().endswith('.txt')):
                print(f"Skipping unsupported file type: {uploaded_file.name}")
                continue
            
            # Save uploaded file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Extract text based on file type
                if uploaded_file.name.lower().endswith('.pdf'):
                    text = self.extract_text_from_pdf(tmp_path)
                else:  # TXT file
                    text = self.extract_text_from_txt(tmp_path)
                
                print(f"Extracted {len(text)} characters from {uploaded_file.name}")
                
                # Smart chunking
                chunk_data = self.smart_chunk_text(text, uploaded_file.name)
                print(f"Created {len(chunk_data)} chunks from {uploaded_file.name}")
                
                # Process chunks in batches to avoid memory issues
                batch_size = 50
                for i in range(0, len(chunk_data), batch_size):
                    batch_chunks = chunk_data[i:i + batch_size]
                    batch_texts = [chunk[0] for chunk in batch_chunks]
                    batch_metadatas = [chunk[1] for chunk in batch_chunks]
                    
                    # Generate embeddings for the batch
                    try:
                        batch_embeddings = self.embedding_model.encode(batch_texts).tolist()
                    except Exception as e:
                        raise Exception(f"Error generating embeddings: {str(e)}")
                    
                    # Create IDs for the batch
                    batch_ids = [str(uuid.uuid4()) for _ in batch_texts]
                    
                    # Add to our collections
                    all_chunks.extend(batch_texts)
                    all_metadatas.extend(batch_metadatas)
                    all_ids.extend(batch_ids)
                    all_embeddings.extend(batch_embeddings)
                    
                processed_files += 1
                    
            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {str(e)}")
                # Don't fail entire batch for one file - continue with others
                continue
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass  # Ignore cleanup errors
        
        # Check if we processed any files successfully
        if processed_files == 0:
            raise Exception("No files were successfully processed. Please check file formats and try again.")
        
        # Add to vector database in batches to avoid timeout
        if all_chunks:
            print(f"Adding {len(all_chunks)} total chunks to vector database...")
            
            add_batch_size = 100
            for i in range(0, len(all_chunks), add_batch_size):
                end_idx = min(i + add_batch_size, len(all_chunks))
                
                batch_embeddings = all_embeddings[i:end_idx]
                batch_documents = all_chunks[i:end_idx]
                batch_metadatas = all_metadatas[i:end_idx]
                batch_ids = all_ids[i:end_idx]
                
                try:
                    self.collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                except Exception as e:
                    raise Exception(f"Error adding batch to vector database: {str(e)}")
                
                print(f"Added batch {i//add_batch_size + 1}/{(len(all_chunks)-1)//add_batch_size + 1}")
        
        print(f"✅ Successfully processed {processed_files}/{total_files} files with {len(all_chunks)} total chunks")
        return self.collection

def process_documents(uploaded_files):
    """Main function to process documents with top-level error handling"""
    try:
        processor = DocumentProcessor()
        return processor.process_documents(uploaded_files)
    except Exception as e:
        # Log the full error for debugging
        print(f"Document processing error: {str(e)}")
        print(traceback.format_exc())
        raise Exception(f"Failed to process documents: {str(e)}")