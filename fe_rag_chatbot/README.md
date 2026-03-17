# RAG Chatbot - WPF Frontend

A modern WPF desktop client for the RAG Enterprise Chatbot system.

## 🛠️ Requirements

- .NET 8.0 SDK
- Windows 10/11

## 🚀 Quick Start

```bash
# 1. Start backend API first
cd ..\rag_chatbot
python scripts\run_server.py

# 2. In a new terminal, start frontend
cd ..\fe_rag_chatbot
dotnet restore
dotnet build
dotnet run --project RAGChatbot.csproj
```

## ⚙️ Configuration

Edit `appsettings.json` to configure the API endpoint:

```json
{
  "ApiSettings": {
    "BaseUrl": "http://localhost:8000/api/v1",
    "TimeoutSeconds": 120
  }
}
```

## 📁 Project Structure

```
fe_rag_chatbot/
├── App.xaml              # Application resources & themes
├── MainWindow.xaml       # Main chat interface
├── RAGChatbot.csproj     # Project file
├── appsettings.json      # API configuration
├── ViewModels/
│   └── MainViewModel.cs  # MVVM ViewModel
├── Models/
│   └── Models.cs         # Data models
├── Services/
│   ├── ChatService.cs    # Chat API client (SSE streaming)
│   └── DocumentService.cs # Document upload/management
└── Converters/
    └── Converters.cs     # XAML value converters
```

## ✨ Features

- **Real-time Streaming**: Server-Sent Events (SSE) for token-by-token response
- **Document Management**: Upload, view, and delete documents
- **Source Citations**: View which documents were used to answer
- **Conversation History**: Multi-turn chat support
- **Material Design**: Modern UI with MaterialDesign theme

## 🔗 Backend

Make sure the RAG Chatbot backend is running at the configured URL before starting the frontend.

```bash
# In the rag_chatbot folder
cd ../rag_chatbot
python scripts/run_server.py
```

If the backend runs on a different host or port, update `BaseUrl` in `appsettings.json` before launching the desktop app.
