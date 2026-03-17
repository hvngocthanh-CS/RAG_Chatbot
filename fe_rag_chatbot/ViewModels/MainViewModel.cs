using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using RAGChatbot.Models;
using RAGChatbot.Services;

namespace RAGChatbot.ViewModels
{
    /// <summary>
    /// Main ViewModel for the RAG Chatbot application.
    /// </summary>
    public partial class MainViewModel : ObservableObject
    {
        private readonly IChatService _chatService;
        private readonly IDocumentService _documentService;
        private string? _conversationId;

        [ObservableProperty]
        private string _inputText = string.Empty;

        [ObservableProperty]
        private bool _isLoading;

        [ObservableProperty]
        private string _statusMessage = "Ready";

        [ObservableProperty]
        private Brush _statusColor = Brushes.Gray;

        [ObservableProperty]
        private DocumentInfo? _selectedDocument;

        public ObservableCollection<ChatMessage> Messages { get; } = new();
        public ObservableCollection<DocumentInfo> Documents { get; } = new();

        public bool CanSend => !string.IsNullOrWhiteSpace(InputText) && !IsLoading;

        public MainViewModel()
        {
            _chatService = ServiceLocator.GetService<IChatService>();
            _documentService = ServiceLocator.GetService<IDocumentService>();

            // Load documents on startup
            _ = LoadDocumentsAsync();
        }

        [RelayCommand]
        private async Task SendMessageAsync()
        {
            if (string.IsNullOrWhiteSpace(InputText) || IsLoading)
                return;

            var question = InputText;
            InputText = string.Empty;

            // Add user message
            Messages.Add(new ChatMessage
            {
                Role = "user",
                Content = question
            });

            IsLoading = true;
            StatusMessage = "Processing...";
            StatusColor = Brushes.Orange;

            try
            {
                // Create a placeholder for streaming response
                var assistantMessage = new ChatMessage
                {
                    Role = "assistant",
                    Content = "",
                    Sources = new ObservableCollection<SourceChunk>()
                };
                Messages.Add(assistantMessage);

                bool gotToken = false;
                string? errorMsg = null;
                await foreach (var chunk in _chatService.ChatStreamAsync(question, _conversationId))
                {
                    if (chunk.Type == "sources")
                    {
                        _conversationId = chunk.ConversationId;
                        foreach (var source in chunk.Sources ?? Enumerable.Empty<SourceChunk>())
                        {
                            assistantMessage.Sources.Add(source);
                        }
                    }
                    else if (chunk.Type == "token")
                    {
                        assistantMessage.Content += chunk.Content;
                        gotToken = true;
                    }
                    else if (chunk.Type == "done")
                    {
                        // Handle error from BE if present
                        if (!string.IsNullOrEmpty(chunk.Error))
                        {
                            errorMsg = chunk.Error;
                        }
                        break;
                    }
                }

                // Always show assistant message, even if error or no token
                if (!gotToken)
                {
                    assistantMessage.Content = errorMsg ?? "Không có câu trả lời phù hợp.";
                    StatusMessage = errorMsg ?? "Không có câu trả lời phù hợp.";
                    StatusColor = Brushes.Red;
                }
                else
                {
                    StatusMessage = "Ready";
                    StatusColor = Brushes.Green;
                }
                // Ensure UI updates for error
                if (!gotToken && string.IsNullOrEmpty(assistantMessage.Content))
                {
                    assistantMessage.Content = "Không có câu trả lời phù hợp hoặc có lỗi từ hệ thống.";
                }
            }
            catch (Exception ex)
            {
                StatusMessage = $"Error: {ex.Message}";
                StatusColor = Brushes.Red;

                Messages.Add(new ChatMessage
                {
                    Role = "assistant",
                    Content = $"Sorry, an error occurred: {ex.Message}"
                });
            }
            finally
            {
                IsLoading = false;
            }
        }

        [RelayCommand]
        private async Task UploadDocumentAsync()
        {
            var dialog = new OpenFileDialog
            {
                Filter = "Documents|*.pdf;*.docx;*.txt;*.md|All Files|*.*",
                Multiselect = true,
                Title = "Select Documents to Upload"
            };

            if (dialog.ShowDialog() == true)
            {
                foreach (var filePath in dialog.FileNames)
                {
                    try
                    {
                        StatusMessage = $"Uploading {Path.GetFileName(filePath)}...";
                        StatusColor = Brushes.Orange;

                        await _documentService.UploadDocumentAsync(filePath);

                        StatusMessage = $"Uploaded {Path.GetFileName(filePath)}";
                        StatusColor = Brushes.Green;
                    }
                    catch (Exception ex)
                    {
                        StatusMessage = $"Failed to upload: {ex.Message}";
                        StatusColor = Brushes.Red;
                    }
                }

                // Refresh document list
                await LoadDocumentsAsync();
            }
        }

        [RelayCommand]
        private async Task DeleteDocumentAsync(string documentId)
        {
            var result = MessageBox.Show(
                "Are you sure you want to delete this document?",
                "Confirm Delete",
                MessageBoxButton.YesNo,
                MessageBoxImage.Warning);

            if (result == MessageBoxResult.Yes)
            {
                try
                {
                    await _documentService.DeleteDocumentAsync(documentId);
                    await LoadDocumentsAsync();
                    StatusMessage = "Document deleted";
                    StatusColor = Brushes.Green;
                }
                catch (Exception ex)
                {
                    StatusMessage = $"Delete failed: {ex.Message}";
                    StatusColor = Brushes.Red;
                }
            }
        }

        [RelayCommand]
        private void ClearChat()
        {
            Messages.Clear();
            _conversationId = null;
            StatusMessage = "Chat cleared";
        }

        private async Task LoadDocumentsAsync()
        {
            try
            {
                var documents = await _documentService.GetDocumentsAsync();
                
                Documents.Clear();
                foreach (var doc in documents)
                {
                    Documents.Add(doc);
                }

                StatusMessage = $"{Documents.Count} documents indexed";
                StatusColor = Brushes.Gray;
            }
            catch (Exception ex)
            {
                StatusMessage = $"Failed to load documents: {ex.Message}";
                StatusColor = Brushes.Red;
            }
        }

        partial void OnInputTextChanged(string value)
        {
            OnPropertyChanged(nameof(CanSend));
        }

        partial void OnIsLoadingChanged(bool value)
        {
            OnPropertyChanged(nameof(CanSend));
        }
    }
}
