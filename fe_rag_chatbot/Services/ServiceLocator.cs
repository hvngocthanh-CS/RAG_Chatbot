using System.Net.Http;
using Microsoft.Extensions.Configuration;

namespace RAGChatbot.Services
{
    /// <summary>
    /// Simple service locator for dependency injection.
    /// In a larger application, consider using a proper DI container.
    /// </summary>
    public static class ServiceLocator
    {
        private static readonly Dictionary<Type, object> _services = new();
        private static bool _initialized;

        public static void Initialize()
        {
            if (_initialized)
                return;

            // Load configuration
            var configuration = new ConfigurationBuilder()
                .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
                .AddJsonFile("appsettings.json", optional: true)
                .Build();

            var apiBaseUrl = configuration["ApiSettings:BaseUrl"] ?? "http://localhost:8000/api/v1";
            var timeout = int.Parse(configuration["ApiSettings:TimeoutSeconds"] ?? "120");

            // Create HttpClient with timeout
            var httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(timeout)
            };

            // Register services
            _services[typeof(IChatService)] = new ChatService(httpClient, apiBaseUrl);
            _services[typeof(IDocumentService)] = new DocumentService(httpClient, apiBaseUrl);

            _initialized = true;
        }

        public static T GetService<T>() where T : class
        {
            if (!_initialized)
                throw new InvalidOperationException("ServiceLocator not initialized. Call Initialize() first.");

            if (_services.TryGetValue(typeof(T), out var service))
                return (T)service;

            throw new InvalidOperationException($"Service {typeof(T).Name} not registered.");
        }
    }
}
