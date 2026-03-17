using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using Newtonsoft.Json;

namespace RAGChatbot.Models
{
    /// <summary>
    /// Represents a chat message in the conversation.
    /// </summary>
    public class ChatMessage : INotifyPropertyChanged
    {
        private string _role = string.Empty;
        private string _content = string.Empty;
        private ObservableCollection<SourceChunk>? _sources;

        public event PropertyChangedEventHandler? PropertyChanged;

        private void NotifyPropertyChanged([CallerMemberName] string propertyName = "")
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        [JsonProperty("role")]
        public string Role
        {
            get => _role;
            set { if (_role != value) { _role = value; NotifyPropertyChanged(); } }
        }

        [JsonProperty("content")]
        public string Content
        {
            get => _content;
            set { if (_content != value) { _content = value; NotifyPropertyChanged(); } }
        }

        [JsonProperty("sources")]
        public ObservableCollection<SourceChunk>? Sources
        {
            get => _sources;
            set { if (_sources != value) { _sources = value; NotifyPropertyChanged(); } }
        }
    }

    /// <summary>
    /// Represents a source chunk from document retrieval.
    /// </summary>
    public class SourceChunk
    {
        [JsonProperty("content")]
        public string Content { get; set; } = string.Empty;
        
        [JsonProperty("document_id")]
        public string DocumentId { get; set; } = string.Empty;
        
        [JsonProperty("document_name")]
        public string DocumentName { get; set; } = string.Empty;
        
        [JsonProperty("page_number")]
        public int? PageNumber { get; set; }
        
        [JsonProperty("chunk_type")]
        public string ChunkType { get; set; } = "text";
        
        [JsonProperty("relevance_score")]
        public double RelevanceScore { get; set; }
    }

    /// <summary>
    /// Represents document metadata.
    /// </summary>
    public class DocumentInfo
    {
        [JsonProperty("id")]
        public string Id { get; set; } = string.Empty;
        
        [JsonProperty("filename")]
        public string Filename { get; set; } = string.Empty;
        
        [JsonProperty("file_type")]
        public string FileType { get; set; } = string.Empty;
        
        [JsonProperty("file_size")]
        public long FileSize { get; set; }
        
        [JsonProperty("upload_date")]
        public string UploadDate { get; set; } = string.Empty;
        
        [JsonProperty("chunks_count")]
        public int ChunksCount { get; set; }
        
        [JsonProperty("status")]
        public string Status { get; set; } = string.Empty;
        
        [JsonProperty("department")]
        public string? Department { get; set; }
        
        [JsonProperty("tags")]
        public List<string> Tags { get; set; } = new();
    }

    /// <summary>
    /// Represents a streaming response chunk.
    /// </summary>
    public class StreamChunk
    {
        [JsonProperty("type")]
        public string Type { get; set; } = string.Empty;
        
        [JsonProperty("content")]
        public string? Content { get; set; }
        
        [JsonProperty("conversation_id")]
        public string? ConversationId { get; set; }
        
        [JsonProperty("sources")]
        public List<SourceChunk>? Sources { get; set; }

        [JsonProperty("error")]
        public string? Error { get; set; }
    }

    /// <summary>
    /// Represents the chat API request.
    /// </summary>
    public class ChatRequest
    {
        [JsonProperty("question")]
        public string Question { get; set; } = string.Empty;
        
        [JsonProperty("conversation_id")]
        public string? ConversationId { get; set; }
        
        [JsonProperty("stream")]
        public bool Stream { get; set; } = true;
        
        [JsonProperty("filters")]
        public Dictionary<string, object>? Filters { get; set; }
    }

    /// <summary>
    /// Represents document upload response.
    /// </summary>
    public class UploadResponse
    {
        [JsonProperty("document_id")]
        public string DocumentId { get; set; } = string.Empty;
        
        [JsonProperty("filename")]
        public string Filename { get; set; } = string.Empty;
        
        [JsonProperty("status")]
        public string Status { get; set; } = string.Empty;
        
        [JsonProperty("message")]
        public string Message { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents document list response.
    /// </summary>
    public class DocumentListResponse
    {
        [JsonProperty("documents")]
        public List<DocumentInfo> Documents { get; set; } = new();
        
        [JsonProperty("total")]
        public int Total { get; set; }
    }
}
