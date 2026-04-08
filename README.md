# RAG Chatbot with Chat History

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and various AI models. This application allows users to upload PDF documents, create vector embeddings, and engage in conversational AI interactions with chat history support.

## Features

- **PDF Document Upload**: Upload and process PDF files for content analysis
- **Vector Embeddings**: Create embeddings using HuggingFace models for semantic search
- **Conversational AI**: Chat with your documents using various LLM providers (Groq, etc.)
- **Chat History**: Maintain conversation context with persistent chat history
- **Configurable Parameters**: Adjust chunk size and overlap for text splitting
- **Multiple Model Support**: Choose from different chat and embedding models
- **Real-time Interaction**: Streamlit-based web interface for easy access

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: Framework for building LLM applications
- **FAISS**: Vector database for similarity search
- **HuggingFace**: Embedding models and API
- **Groq**: LLM provider for chat functionality
- **PyMuPDF**: PDF processing library

## Installation

### Prerequisites

- Python 3.14 or higher
- API keys for:
  - Groq (for chat models)
  - HuggingFace (for embedding models)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SambhavSurthi/RAG-Chatbot.git
   cd RAG-Chatbot
   ```

2. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

   Or using uv (if available):
   ```bash
   uv pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Configure Models**:
   - Select your preferred chat model from the sidebar
   - Enter your Groq API key
   - Choose an embedding model
   - Enter your HuggingFace API token

2. **Upload Document**:
   - Upload a PDF file using the file uploader
   - The system will process the document and create embeddings

3. **Chat Interface**:
   - Enter your questions in the chat input
   - The chatbot will retrieve relevant information from the uploaded document
   - Chat history is maintained throughout the session

4. **Additional Controls**:
   - Adjust chunk size and overlap for text processing
   - Clear chat history when needed
   - Delete uploaded documents

## Project Structure

```
RAG-Chatbot/
├── main.py                 # Main Streamlit application
├── pyproject.toml          # Project configuration
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── raw_pdf/               # Directory for uploaded PDFs
├── .gitignore             # Git ignore file
├── .python-version        # Python version specification
└── uv.lock                # Dependency lock file
```

## Configuration

### Environment Variables

You can set API keys using environment variables:

```bash
export GROQ_API_KEY="your-groq-api-key"
export HUGGINGFACE_API_TOKEN="your-huggingface-token"
```

### Model Options

**Chat Models**:
- `qwen/qwen3-32b`
- `groq/compound`
- `llama-3.1-8b-instant`
- `openai/gpt-oss-120b`

**Embedding Models**:
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`
- `microsoft/harrier-oss-v1-0.6b`

## How It Works

1. **Document Processing**: PDFs are loaded and split into manageable chunks
2. **Embedding Creation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in FAISS for efficient retrieval
4. **Query Processing**: User queries are embedded and matched against document vectors
5. **Context Retrieval**: Relevant document sections are retrieved as context
6. **Response Generation**: LLM generates responses using retrieved context and chat history

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://www.langchain.com/) for the LLM framework
- [Streamlit](https://streamlit.io/) for the web interface
- [HuggingFace](https://huggingface.co/) for embedding models
- [Groq](https://groq.com/) for LLM services

## Contact

Sambhav Surthi - [GitHub](https://github.com/SambhavSurthi)

Project Link: [https://github.com/SambhavSurthi/RAG-Chatbot](https://github.com/SambhavSurthi/RAG-Chatbot)