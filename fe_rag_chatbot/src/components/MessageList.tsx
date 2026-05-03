import React, { useState, useRef, useEffect } from 'react';
import { Message, Source } from '../types';

interface MessageListProps {
  messages: Message[];
  isLoading: boolean;
  streamingContent: string;
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const score = Math.round(source.relevance_score * 100);

  return (
    <div className="border border-gray-200 rounded-md overflow-hidden text-xs">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="shrink-0 w-5 h-5 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-semibold">
            {index + 1}
          </span>
          <span className="truncate text-gray-700 font-medium">{source.document_name}</span>
          {source.page_number != null && (
            <span className="shrink-0 text-gray-400">p.{source.page_number}</span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0 ml-2">
          <span
            className={`px-1.5 py-0.5 rounded font-semibold ${
              score >= 80
                ? 'bg-green-100 text-green-700'
                : score >= 60
                ? 'bg-yellow-100 text-yellow-700'
                : 'bg-red-100 text-red-700'
            }`}
          >
            {score}%
          </span>
          <span className="text-gray-400">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>
      {expanded && (
        <div className="px-3 py-2 bg-white text-gray-600 leading-relaxed border-t border-gray-100">
          {source.content}
        </div>
      )}
    </div>
  );
}

function SourcesSection({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-700 transition-colors"
      >
        <span>{open ? '▼' : '▶'}</span>
        <span>{sources.length} nguồn tài liệu</span>
      </button>
      {open && (
        <div className="mt-2 flex flex-col gap-1.5">
          {sources.map((src, i) => (
            <SourceCard key={src.document_id + i} source={src} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading,
  streamingContent,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      {messages.length === 0 && !isLoading && (
        <div className="flex items-center justify-center h-full">
          <div className="text-center text-gray-400">
            <div className="text-5xl mb-3">💬</div>
            <p className="text-lg font-medium text-gray-500">RAG Chatbot</p>
            <p className="text-sm mt-1">
              Upload tài liệu rồi đặt câu hỏi. Chatbot sẽ trả lời dựa trên nội dung tài liệu.
            </p>
          </div>
        </div>
      )}

      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          {message.role === 'assistant' && (
            <div className="w-7 h-7 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 shrink-0">
              AI
            </div>
          )}
          <div className={`max-w-[75%] ${message.role === 'user' ? '' : 'flex-1'}`}>
            <div
              className={`px-4 py-2.5 rounded-2xl ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white rounded-br-sm'
                  : 'bg-gray-100 text-gray-900 rounded-bl-sm'
              }`}
            >
              <p className="text-sm whitespace-pre-wrap break-words leading-relaxed">
                {message.content}
              </p>
              <span className="text-xs opacity-50 mt-1 block">
                {message.timestamp.toLocaleTimeString('vi-VN', {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            </div>
            {message.role === 'assistant' && message.sources && (
              <SourcesSection sources={message.sources} />
            )}
          </div>
        </div>
      ))}

      {isLoading && streamingContent && (
        <div className="flex justify-start">
          <div className="w-7 h-7 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 shrink-0">
            AI
          </div>
          <div className="max-w-[75%] flex-1 px-4 py-2.5 rounded-2xl rounded-bl-sm bg-gray-100 text-gray-900">
            <p className="text-sm whitespace-pre-wrap break-words leading-relaxed">
              {streamingContent}
              <span className="inline-block w-0.5 h-4 bg-gray-500 ml-0.5 animate-pulse" />
            </p>
          </div>
        </div>
      )}

      {isLoading && !streamingContent && (
        <div className="flex justify-start items-center">
          <div className="w-7 h-7 rounded-full bg-blue-500 flex items-center justify-center text-white text-xs font-bold mr-2 shrink-0">
            AI
          </div>
          <div className="bg-gray-100 rounded-2xl rounded-bl-sm px-4 py-3">
            <div className="flex space-x-1.5">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
};
