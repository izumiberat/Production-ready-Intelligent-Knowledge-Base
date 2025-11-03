import openai
from sentence_transformers import SentenceTransformer
import chromadb
from typing import Tuple, List, Dict, Any
import os
import re

class QAEngine:
    def __init__(self):
        # Use the same embedding model as document processor
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize OpenAI client with error handling
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise Exception("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
    
    def find_relevant_chunks(self, question: str, collection: chromadb.Collection, n_results: int = 5) -> Tuple[List[str], List[Dict]]:
        """Enhanced semantic search with query expansion"""
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_model.encode([question]).tolist()
            
            # Search in vector database with better parameters
            results = collection.query(
                query_embeddings=question_embedding,
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            # Filter by similarity threshold (cosine distance)
            filtered_docs = []
            filtered_metas = []
            
            for doc, meta, distance in zip(documents, metadatas, distances):
                # Convert cosine distance to similarity score (1 - distance)
                similarity = 1 - distance
                if similarity > 0.3:  # Reasonable threshold
                    filtered_docs.append(doc)
                    # Add similarity score to metadata
                    meta['similarity_score'] = round(similarity, 3)
                    filtered_metas.append(meta)
            
            return filtered_docs, filtered_metas
            
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")
    
    def generate_answer(self, question: str, context_chunks: List[str], sources: List[Dict]) -> Tuple[str, List[Dict]]:
        """Enhanced answer generation with better prompting and citation"""
        if not context_chunks:
            return "I couldn't find enough relevant information in the documents to answer this question. Please try rephrasing your question or adding more relevant documents.", []

        try:
            # Prepare context with source information
            context_parts = []
            unique_sources = set()
            
            for i, (chunk, source_meta) in enumerate(zip(context_chunks, sources)):
                source_name = source_meta.get('source', 'Unknown document')
                similarity = source_meta.get('similarity_score', 'N/A')
                
                context_parts.append(f"[Source: {source_name} | Relevance: {similarity}]\n{chunk}")
                unique_sources.add(source_name)
            
            context = "\n\n".join(context_parts)
            
            # Enhanced prompt for better answers
            prompt = f"""You are an expert research assistant. Based EXCLUSIVELY on the provided context documents, answer the user's question.

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the provided context. Do not use external knowledge.
2. If the context doesn't contain enough information to fully answer, say so and indicate what information is missing.
3. Be specific and cite your sources using the source names provided.
4. If different sources conflict, acknowledge the conflict and present both viewpoints.
5. Keep the answer comprehensive but concise.

STRUCTURE YOUR ANSWER:
- Start with a direct answer to the question
- Provide supporting evidence from the sources
- Clearly cite which source each piece of information came from
- End with a summary of the key findings"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise research assistant that provides accurate, source-cited answers based only on the provided documents."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.1,  # Low temperature for factual accuracy
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Format the sources for display
            formatted_sources = []
            for source in sources:
                source_name = source.get('source', 'Unknown')
                similarity = source.get('similarity_score', 'N/A')
                formatted_sources.append(f"{source_name} (relevance: {similarity})")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_formatted_sources = []
            for src in formatted_sources:
                if src not in seen:
                    seen.add(src)
                    unique_formatted_sources.append(src)
            
            return answer, unique_formatted_sources
            
        except openai.APIError as e:
            raise Exception(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")

def get_answer(question: str, vector_store):
    """Main function to get answer for a question"""
    if not vector_store:
        raise Exception("No documents processed yet. Please upload and process documents first.")
    
    engine = QAEngine()
    
    # Find relevant chunks
    relevant_chunks, sources = engine.find_relevant_chunks(question, vector_store, n_results=5)
    
    # Generate answer
    answer, cited_sources = engine.generate_answer(question, relevant_chunks, sources)
    
    return answer, cited_sources