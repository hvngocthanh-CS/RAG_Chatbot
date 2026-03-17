using System.Windows;
using System.Windows.Controls;

namespace RAGChatbot
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            
            // Auto-scroll chat to bottom when new messages arrive or content changes (streaming)
            if (DataContext is ViewModels.MainViewModel viewModel)
            {
                viewModel.Messages.CollectionChanged += (s, e) =>
                {
                    // Subscribe to PropertyChanged on newly added messages
                    if (e.NewItems != null)
                    {
                        foreach (Models.ChatMessage msg in e.NewItems)
                        {
                            msg.PropertyChanged += (_, _) =>
                                Dispatcher.InvokeAsync(() => ChatScrollViewer.ScrollToEnd());
                        }
                    }

                    Dispatcher.InvokeAsync(() => ChatScrollViewer.ScrollToEnd());
                };
            }
        }
    }
}
