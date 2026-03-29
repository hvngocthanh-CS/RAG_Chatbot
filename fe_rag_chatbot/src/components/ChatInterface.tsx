import React from 'react';
import { MessageList } from './MessageList';
import { InputBox } from './InputBox';
import { DocumentUpload } from './DocumentUpload';
import { Message } from '../types';

interface ChatInterfaceProps {
  messages: Message[];
  isLoading: boolean;
  streamingContent: string;
  onSendMessage: (message: string) => Promise<void>;
  onUploadComplete: (message: string) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  isLoading,
  streamingContent,
  onSendMessage,
  onUploadComplete,
}) => {
  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-4 flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold">RAG Chatbot</h1>
          <p className="text-sm opacity-90">Retrieval-Augmented Generation</p>
        </div>
        <DocumentUpload onUploadComplete={onUploadComplete} />
      </div>

      {/* Messages */}
      <MessageList
        messages={messages}
        isLoading={isLoading}
        streamingContent={streamingContent}
      />

      {/* Input */}
      <InputBox onSendMessage={onSendMessage} isLoading={isLoading} />
    </div>
  );
};
