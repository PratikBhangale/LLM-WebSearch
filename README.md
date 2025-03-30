# AI Chat with Web Search

A Streamlit-powered chatbot that combines Groq's LLMs with Tavily's real-time web search for intelligent, context-aware conversations.

## 🌟 Features

- **Advanced LLM Integration**
  - Multiple Groq model options (llama3-70b-8192, llama3-8b-8192, mixtral-8x7b-32768)
  - Real-time token streaming
  - Temperature and context control

- **Smart Web Search**
  - Context-aware query generation
  - Real-time web search via Tavily API
  - Intelligent result incorporation

- **Interactive Interface**
  - Clean Streamlit-based UI
  - Expandable search details
  - Model information display
  - Chat history management

## 🚀 Getting Started

### Prerequisites
```bash
- Python 3.8+
- Groq API key
- Tavily API key
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-chat-with-web-search.git
cd ai-chat-with-web-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Running the App
```bash
streamlit run test.py
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Provider**: Groq
- **Search API**: Tavily
- **Framework**: LangChain
- **Language**: Python

## 📝 Usage

1. Select your preferred Groq model from the sidebar
2. Toggle web search functionality
3. Start chatting with context-aware responses
4. View search details and model information in expandable sections
5. Clear chat history as needed

## ⚙️ Configuration

Modify model parameters in the sidebar:
- Model selection
- Web search toggle
- Chat history management

## 📄 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a pull request

## 💬 Support

For issues and feature requests, please open an issue on GitHub.
