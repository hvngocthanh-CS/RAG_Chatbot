using System.IO;
using System.Net.Http;
using System.Net.Http.Json;
using RAGChatbot.Models;

namespace RAGChatbot.Services
{
    /// <summary>
    /// Implementation of document service for uploading and managing documents.
    /// </summary>
    public class DocumentService : IDocumentService
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public DocumentService(HttpClient httpClient, string baseUrl)
        {
            _httpClient = httpClient;
            _baseUrl = baseUrl.TrimEnd('/');
        }

        public async Task<UploadResponse> UploadDocumentAsync(
            string filePath, 
            string? department = null, 
            List<string>? tags = null)
        {
            using var fileStream = File.OpenRead(filePath);
            using var content = new MultipartFormDataContent();
            
            var fileContent = new StreamContent(fileStream);
            content.Add(fileContent, "file", Path.GetFileName(filePath));

            var url = $"{_baseUrl}/documents/upload";
            
            // Add query parameters
            var queryParams = new List<string>();
            if (!string.IsNullOrEmpty(department))
                queryParams.Add($"department={Uri.EscapeDataString(department)}");
            if (tags != null && tags.Any())
                queryParams.Add($"tags={Uri.EscapeDataString(string.Join(",", tags))}");
            
            if (queryParams.Any())
                url += "?" + string.Join("&", queryParams);

            var response = await _httpClient.PostAsync(url, content);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<UploadResponse>();
            return result ?? new UploadResponse { Status = "unknown" };
        }

        public async Task<List<DocumentInfo>> GetDocumentsAsync()
        {
            var response = await _httpClient.GetFromJsonAsync<DocumentListResponse>(
                $"{_baseUrl}/documents");
            
            return response?.Documents ?? new List<DocumentInfo>();
        }

        public async Task DeleteDocumentAsync(string documentId)
        {
            var response = await _httpClient.DeleteAsync($"{_baseUrl}/documents/{documentId}");
            response.EnsureSuccessStatusCode();
        }
    }
}
