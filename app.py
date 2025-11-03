import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import sys
from pathlib import Path
from datetime import datetime
import time

# Add the lib directory to the path
sys.path.append(str(Path(__file__).parent / "lib"))

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Intelligent Knowledge Base - AI Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .chat-question {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
    }
    .chat-answer {
        background-color: #f3e5f5;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 4px solid #9C27B0;
    }
    .source-item {
        background-color: #e8f5e8;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border-left: 3px solid #4CAF50;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .stButton button {
        width: 100%;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    default_state = {
        'documents_processed': False,
        'vector_store': None,
        'processed_docs': [],
        'processing_log': [],
        'chat_history': [],
        'processing_start_time': None,
        'last_question_time': None,
        'total_questions': 0
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def log_message(message: str, message_type: str = "info"):
    """Add message to processing log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icon = {
        "info": "🔵",
        "success": "✅", 
        "warning": "⚠️",
        "error": "❌"
    }.get(message_type, "🔵")
    
    st.session_state.processing_log.append(f"{icon} [{timestamp}] {message}")

def clear_documents():
    """Clear all documents and reset state"""
    st.session_state.documents_processed = False
    st.session_state.vector_store = None
    st.session_state.processed_docs = []
    st.session_state.chat_history = []
    st.session_state.total_questions = 0
    log_message("Documents and chat history cleared", "info")

def display_metrics():
    """Display system metrics"""
    if st.session_state.documents_processed:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📄</h3>
                <h4>{len(st.session_state.processed_docs)}</h4>
                <p>Documents Loaded</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💬</h3>
                <h4>{st.session_state.total_questions}</h4>
                <p>Questions Asked</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_time = "N/A"
            if st.session_state.chat_history:
                times = [float(chat['processing_time'].rstrip('s')) for chat in st.session_state.chat_history]
                avg_time = f"{sum(times)/len(times):.1f}s"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡</h3>
                <h4>{avg_time}</h4>
                <p>Avg Response Time</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    init_session_state()
    
    # Header
    st.markdown("<h1 class='main-header'>📚 Intelligent Knowledge Base</h1>", 
                unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h3>AI-Powered Document Q&A with Source Citations</h3>
        <p>Upload your documents and get instant, accurate answers with verified sources</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
        st.header("📄 Document Management")
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Supported: PDF documents, Text files"
        )
        
        if uploaded_files:
            st.success(f"📎 {len(uploaded_files)} file(s) ready for processing")
            
            # File details
            with st.expander("📋 File Details", expanded=True):
                total_size = 0
                for file in uploaded_files:
                    file_size_kb = file.size // 1024
                    total_size += file_size_kb
                    st.write(f"• {file.name} ({file_size_kb} KB)")
                st.write(f"**Total size:** {total_size} KB")
            
            # Process button
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                st.session_state.processing_start_time = time.time()
                
                with st.spinner("Processing documents... This may take a few moments."):
                    try:
                        from document_processor import process_documents
                        
                        # Clear previous state
                        st.session_state.vector_store = None
                        st.session_state.documents_processed = False
                        st.session_state.processing_log = []
                        
                        # Process documents
                        st.session_state.vector_store = process_documents(uploaded_files)
                        st.session_state.documents_processed = True
                        st.session_state.processed_docs = [f.name for f in uploaded_files]
                        
                        # Calculate processing time
                        processing_time = time.time() - st.session_state.processing_start_time
                        
                        st.success(f"✅ Processing complete! Time: {processing_time:.1f}s")
                        log_message(f"Processed {len(uploaded_files)} files in {processing_time:.1f}s", "success")
                        
                    except Exception as e:
                        error_msg = f"Error processing documents: {str(e)}"
                        st.error(f"❌ {error_msg}")
                        log_message(error_msg, "error")
        
        # System status
        st.markdown("---")
        st.subheader("📊 System Status")
        
        if st.session_state.documents_processed:
            st.success("✅ System Ready")
            
            # Quick actions
            st.subheader("🔧 Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.session_state.total_questions = 0
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    clear_documents()
                    st.rerun()
        else:
            st.warning("⏳ Waiting for Documents")
            st.info("Upload files and click 'Process Documents' to begin.")
        
        # Processing log
        if st.session_state.processing_log:
            with st.expander("📝 Processing Log", expanded=False):
                for log_entry in st.session_state.processing_log[-6:]:
                    st.text(log_entry)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            <p>Intelligent Knowledge Base v1.0</p>
            <p>Built with Streamlit + OpenAI + ChromaDB</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Q&A Section
        st.header("💬 Ask Questions")
        
        if st.session_state.documents_processed:
            # Display metrics
            display_metrics()
            
            # Question input
            question = st.text_area(
                "Your question:",
                placeholder="Example: What are the main objectives mentioned in the documents?",
                height=120,
                key="question_input"
            )
            
            # Ask button
            col1a, col1b = st.columns([3, 1])
            with col1b:
                ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
            
            if ask_button and question:
                st.session_state.last_question_time = time.time()
                st.session_state.total_questions += 1
                
                with st.spinner("🔍 Searching documents and generating answer..."):
                    try:
                        from qa_engine import get_answer
                        
                        # Get answer
                        start_time = time.time()
                        answer, sources = get_answer(question, st.session_state.vector_store)
                        processing_time = time.time() - start_time
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": answer,
                            "sources": sources,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "processing_time": f"{processing_time:.1f}s"
                        })
                        
                        # Display answer
                        st.markdown(f"""
                        <div class="chat-question">
                            <strong>🤔 Your Question:</strong><br>
                            {question}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="chat-answer">
                            <strong>🤖 AI Answer:</strong><br>
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if sources:
                            st.subheader("📚 Source Citations")
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"<div class='source-item'>{i}. {source}</div>", 
                                           unsafe_allow_html=True)
                        
                        # Show performance
                        st.success(f"✅ Answer generated in {processing_time:.1f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
            elif ask_button and not question:
                st.warning("Please enter a question first.")
        
        else:
            # Welcome state
            st.markdown("""
            <div class="info-box">
                <h3>👋 Welcome to Intelligent Knowledge Base!</h3>
                <p>Transform your document search experience with AI-powered Q&A.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Features overview
            st.subheader("✨ Key Features")
            
            feat_col1, feat_col2 = st.columns(2)
            
            with feat_col1:
                st.markdown("""
                **📄 Smart Document Processing**
                - PDF & TXT file support
                - Intelligent text extraction
                - Semantic chunking
                - Automatic metadata generation
                """)
                
                st.markdown("""
                **🔍 Advanced Search**
                - Vector-based similarity
                - Semantic understanding
                - Multi-document search
                - Relevance scoring
                """)
            
            with feat_col2:
                st.markdown("""
                **🤖 Intelligent Answers**
                - Natural language responses
                - Source citation
                - Context-aware synthesis
                - Factual accuracy
                """)
                
                st.markdown("""
                **🚀 Enterprise Ready**
                - Production-grade error handling
                - Performance optimization
                - Secure processing
                - Scalable architecture
                """)
            
            # Sample questions
            with st.expander("💡 Sample Questions to Try", expanded=True):
                st.markdown("""
                After uploading documents, try asking:
                
                **Project Documents:**
                - *"What are the main objectives?"*
                - *"What methodology is recommended?"*  
                - *"Who are the key stakeholders?"*
                - *"What risks are identified?"*
                
                **Technical Documents:**
                - *"Explain the architecture overview"*
                - *"What are the system requirements?"*
                - *"How does the authentication work?"*
                
                **Policy Documents:**
                - *"What are the security protocols?"*
                - *"What is the approval process?"*
                - *"What compliance standards apply?"*
                """)
    
    with col2:
        # Dashboard panel
        st.header("📊 Dashboard")
        
        if st.session_state.documents_processed:
            # Loaded documents
            with st.expander("📋 Loaded Documents", expanded=True):
                for doc in st.session_state.processed_docs:
                    st.write(f"• {doc}")
            
            # Quick questions
            with st.expander("🚀 Quick Questions", expanded=True):
                quick_questions = [
                    "What are the main goals?",
                    "What methodology is used?",
                    "Who are the key people?",
                    "What are the risks?",
                    "What are the deliverables?"
                ]
                
                for q in quick_questions:
                    if st.button(q, key=f"quick_{hash(q)}", use_container_width=True):
                        st.session_state.question_input = q
                        st.rerun()
            
            # Chat history
            if st.session_state.chat_history:
                st.subheader("💭 Recent Questions")
                for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
                    with st.expander(f"Q: {chat['question'][:50]}...", expanded=i==0):
                        st.write(f"**A:** {chat['answer'][:150]}...")
                        st.caption(f"⏰ {chat['timestamp']} | ⚡ {chat['processing_time']}")

if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("""
        🔑 OpenAI API Key not found!
        
        Please set your OpenAI API key in one of these ways:
        
        1. **For local development:** Create a `.env` file with:
           ```
           OPENAI_API_KEY=your-api-key-here
           ```
        
        2. **For Streamlit Cloud:** Add to your app secrets in the dashboard
        
        Get your API key from: https://platform.openai.com/api-keys
        """)
    else:
        main()