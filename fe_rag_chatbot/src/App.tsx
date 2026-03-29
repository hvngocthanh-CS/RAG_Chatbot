import { useState, useCallback } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { Message } from './types';
import { chatAPI } from './services/api';
import './App.css';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);

  const handleSendMessage = useCallback(
    async (question: string) => {
      // Add user message
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: question,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      setStreamingContent('');

      try {
        let fullContent = '';

        for await (const chunk of chatAPI.streamChat({
          question,
          conversationId,
          stream: true,
        })) {
          if (chunk.type === 'sources' && chunk.conversation_id) {
            setConversationId(chunk.conversation_id);
          } else if (chunk.type === 'token' && chunk.content) {
            fullContent += chunk.content;
            setStreamingContent(fullContent);
          } else if (chunk.type === 'error') {
            setStreamingContent(`Error: ${chunk.error || 'Unknown error'}`);
            break;
          } else if (chunk.type === 'done') {
            // Message complete
            break;
          }
        }

        // Add assistant message
        if (fullContent) {
          const assistantMessage: Message = {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: fullContent,
            timestamp: new Date(),
          };

          setMessages((prev) => [...prev, assistantMessage]);
        }
      } catch (error: any) {
        const errorMessage: Message = {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `Error: ${error.message || 'Failed to get response'}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setIsLoading(false);
        setStreamingContent('');
      }
    },
    [conversationId]
  );

  const handleUploadComplete = (message: string) => {
    const notificationMessage: Message = {
      id: `notification-${Date.now()}`,
      role: 'assistant',
      content: message,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, notificationMessage]);
  };

  return (
    <div className="w-full h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl h-full max-h-[90vh] bg-white rounded-lg shadow-2xl overflow-hidden">
        <ChatInterface
          messages={messages}
          isLoading={isLoading}
          streamingContent={streamingContent}
          onSendMessage={handleSendMessage}
          onUploadComplete={handleUploadComplete}
        />
      </div>
    </div>
  );
}

export default App;
