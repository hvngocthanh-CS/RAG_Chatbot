using RAGChatbot.Models;

namespace RAGChatbot.Services
{
    /// <summary>
    /// Interface for chat operations.
    /// </summary>
    public interface IChatService
    {
        /// <summary>
        /// Stream chat response for a question.
        /// </summary>
        IAsyncEnumerable<StreamChunk> ChatStreamAsync(string question, string? conversationId);
        
        /// <summary>
        /// Get chat response without streaming.
        /// </summary>
        Task<(string Answer, List<SourceChunk> Sources)> ChatAsync(string question, string? conversationId);
    }

    /// <summary>
    /// Interface for document operations.
    /// </summary>
    public interface IDocumentService
    {
        /// <summary>
        /// Upload a document.
        /// </summary>
        Task<UploadResponse> UploadDocumentAsync(string filePath, string? department = null, List<string>? tags = null);
        
        /// <summary>
        /// Get list of indexed documents.
        /// </summary>
        Task<List<DocumentInfo>> GetDocumentsAsync();
        
        /// <summary>
        /// Delete a document.
        /// </summary>
        Task DeleteDocumentAsync(string documentId);
    }
}
