# 🎫 Support Ticket Classifier ChatBot with RAG

A powerful AI-powered support ticket classification and analytics system built with Retrieval-Augmented Generation (RAG) technology. This application helps organizations analyze, categorize, and respond to support tickets efficiently using advanced natural language processing and machine learning techniques.

## 🌟 Features

### 🤖 **Intelligent Chatbot Interface**
- **Multi-format Support**: Upload and process both PDF and CSV files
- **Real-time Chat**: Interactive conversation with uploaded documents
- **Context-aware Responses**: AI understands document content and provides relevant answers
- **Multi-language Support**: Handles various text encodings and formats

### 📊 **Advanced Analytics Dashboard**
- **Priority Analysis**: Visualize ticket distribution by priority levels (Critical, High, Medium, Low)
- **Status Tracking**: Monitor ticket status distribution (Open, Closed, Pending Customer Response)
- **Type Classification**: Analyze different ticket types (Technical Issues, Billing, Refunds, etc.)
- **Satisfaction Metrics**: Track customer satisfaction ratings and trends
- **Resolution Time Analysis**: Histograms and box plots for resolution time patterns
- **Interactive Visualizations**: Bar charts, pie charts, histograms, and correlation heatmaps

### 🔍 **RAG-Powered Document Processing**
- **Vector Embeddings**: Uses HuggingFace BGE embeddings for semantic search
- **Qdrant Vector Database**: Efficient storage and retrieval of document chunks
- **Smart Text Splitting**: Recursive character text splitter for optimal chunk sizes
- **Multi-line CSV Support**: Handles complex CSV files with embedded line breaks

### 🚀 **Modern Tech Stack**
- **LLM Integration**: Groq's Llama-3.3-70B-Versatile model for fast inference
- **Streamlit UI**: Beautiful, responsive web interface
- **LangChain Framework**: Modular AI application development
- **Pandas & Matplotlib**: Data analysis and visualization
- **Seaborn**: Statistical data visualization

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Chatbot Manager │────│  Groq LLM API   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │              ┌────────▼────────┐
         │              │  Qdrant Vector  │
         │              │     Database    │
         │              └─────────────────┘
         │
    ┌────▼────┐    ┌──────────────┐
    │Analytics│────│  Helpdesk    │
    │ Dashboard│    │  Analytics   │
    └─────────┘    └──────────────┘
```

## 📁 Project Structure

```
Support-Ticket-Classifier/
├── 📄 new.py                    # Main Streamlit application
├── 🤖 chatbot.py               # Chatbot manager with RAG integration
├── 📊 helpdesk_analytics.py    # Analytics and visualization engine
├── 🔧 vectors.py               # Vector embeddings and document processing
├── 📋 requirements.txt         # Python dependencies
├── ⚙️ .streamlit/config.toml   # Streamlit configuration
├── 📜 LICENSE                  # MIT License
└── 📖 README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Qdrant vector database running on localhost:6333
- Groq API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Support-Ticket-Classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   ```

4. **Start Qdrant vector database**
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or download and run locally
   # Visit: https://qdrant.tech/documentation/quick-start/
   ```

5. **Run the application**
   ```bash
   streamlit run new.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. **Upload Documents**
- Navigate to the "🤖 Chatbot" section
- Upload your support ticket data (CSV) or documentation (PDF)
- The system will automatically process and create embeddings

### 2. **Generate Embeddings**
- Check the "✅ Generate Embeddings" checkbox
- Wait for the vector database to be created
- You'll see a success message when complete

### 3. **Chat with Documents**
- Use the chat interface to ask questions about your data
- Ask about ticket counts, priorities, statuses, or any specific information
- The AI will provide context-aware responses based on your documents

### 4. **Analytics Dashboard**
- Navigate to "📊 Helpdesk Analytics"
- Use pre-built visualization buttons for common analyses
- Or enter custom queries for specific insights

## 🔧 Configuration

### **Model Settings**
- **Embedding Model**: BAAI/bge-small-en (configurable in `vectors.py`)
- **LLM Model**: llama-3.3-70b-versatile (configurable in `chatbot.py`)
- **Chunk Size**: 1000 characters with 100 character overlap

### **Vector Database**
- **Database**: Qdrant
- **URL**: http://localhost:6333
- **Collection**: vector_db



## 📊 Sample Data Analysis

The system can analyze various aspects of your support ticket data:

### **Priority Distribution**
- Critical: 279 tickets (27.9%)
- Medium: 258 tickets (25.8%)
- Low: 253 tickets (25.3%)
- High: 209 tickets (20.9%)

### **Status Analysis**
- Pending Customer Response: 335 tickets
- Closed: 333 tickets
- Open: 331 tickets

### **Ticket Types**
- Cancellation requests
- Technical issues
- Refund requests
- Billing inquiries
- Product inquiries


## 📈 Performance

- **Embedding Generation**: ~2-5 seconds per 1000 documents
- **Query Response**: <2 seconds for most queries
- **Visualization Generation**: <3 seconds for standard charts
- **Memory Usage**: ~500MB for typical datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


##  Acknowledgments

- **LangChain** for the RAG framework
- **Groq** for fast LLM inference
- **Qdrant** for vector database
- **Streamlit** for the web interface
- **HuggingFace** for embedding models


---

**Built with ❤️ for efficient support ticket management and analysis**
