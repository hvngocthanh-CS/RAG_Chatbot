import { useState, useCallback, useRef } from 'react';
import { ChatInterface } from './components/ChatInterface';
import { Message, Source } from './types';
import { chatAPI } from './services/api';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);

  const sourcesRef = useRef<Source[]>([]);

  const handleSendMessage = useCallback(
    async (question: string) => {
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: question,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      setStreamingContent('');
      sourcesRef.current = [];

      try {
        let fullContent = '';

        for await (const chunk of chatAPI.streamChat({
          question,
          conversationId,
          stream: true,
        })) {
          if (chunk.type === 'sources') {
            if (chunk.conversation_id) setConversationId(chunk.conversation_id);
            if (chunk.sources) sourcesRef.current = chunk.sources;
          } else if (chunk.type === 'token' && chunk.content) {
            fullContent += chunk.content;
            setStreamingContent(fullContent);
          } else if (chunk.type === 'error') {
            setStreamingContent(`Error: ${chunk.error || 'Unknown error'}`);
            break;
          } else if (chunk.type === 'done') {
            break;
          }
        }

        if (fullContent) {
          const assistantMessage: Message = {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: fullContent,
            timestamp: new Date(),
            sources: sourcesRef.current,
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

  const handleNewConversation = useCallback(() => {
    setMessages([]);
    setConversationId(undefined);
    setStreamingContent('');
  }, []);

  return (
    <div className="w-full h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl h-full max-h-[95vh] bg-white rounded-lg shadow-2xl overflow-hidden">
        <ChatInterface
          messages={messages}
          isLoading={isLoading}
          streamingContent={streamingContent}
          onSendMessage={handleSendMessage}
          onUploadComplete={handleUploadComplete}
          onNewConversation={handleNewConversation}
        />
      </div>
    </div>
  );
}

export default App;
