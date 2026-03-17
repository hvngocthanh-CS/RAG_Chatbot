using System.Net.Http;
using System.Net.Http.Json;
using System.IO;
using System.Text;
using System.Text.Json;
using Newtonsoft.Json;
using RAGChatbot.Models;

namespace RAGChatbot.Services
{
    /// <summary>
    /// Implementation of chat service communicating with the RAG API.
    /// </summary>
    public class ChatService : IChatService
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public ChatService(HttpClient httpClient, string baseUrl)
        {
            _httpClient = httpClient;
            _baseUrl = baseUrl.TrimEnd('/');
        }

        public async IAsyncEnumerable<StreamChunk> ChatStreamAsync(string question, string? conversationId)
        {
            var request = new ChatRequest
            {
                Question = question,
                ConversationId = conversationId,
                Stream = true
            };

            var json = JsonConvert.SerializeObject(request);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            using var requestMessage = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/chat")
            {
                Content = content
            };

            using var response = await _httpClient.SendAsync(
                requestMessage, 
                HttpCompletionOption.ResponseHeadersRead);

            response.EnsureSuccessStatusCode();

            using var stream = await response.Content.ReadAsStreamAsync();
            using var reader = new StreamReader(stream);

            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                
                if (string.IsNullOrEmpty(line))
                    continue;

                if (line.StartsWith("data: "))
                {
                    var data = line.Substring(6);
                    
                    if (string.IsNullOrEmpty(data))
                        continue;

                    StreamChunk? chunk = null;
                    try
                    {
                        chunk = JsonConvert.DeserializeObject<StreamChunk>(data);
                    }
                    catch
                    {
                        continue;
                    }

                    if (chunk != null)
                    {
                        yield return chunk;

                        if (chunk.Type == "done")
                            break;
                    }
                }
            }
        }

        public async Task<(string Answer, List<SourceChunk> Sources)> ChatAsync(string question, string? conversationId)
        {
            var request = new ChatRequest
            {
                Question = question,
                ConversationId = conversationId,
                Stream = false
            };

            var response = await _httpClient.PostAsJsonAsync($"{_baseUrl}/chat", request);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<ChatResponse>();
            
            return (result?.Answer ?? "", result?.Sources ?? new List<SourceChunk>());
        }

        private class ChatResponse
        {
            public string Answer { get; set; } = string.Empty;
            public string ConversationId { get; set; } = string.Empty;
            public List<SourceChunk> Sources { get; set; } = new();
            public int ProcessingTimeMs { get; set; }
        }
    }
}
