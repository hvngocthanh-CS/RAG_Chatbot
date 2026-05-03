import React, { useState, useRef, useEffect } from 'react';

interface InputBoxProps {
  onSendMessage: (message: string) => Promise<void>;
  isLoading: boolean;
}

export const InputBox: React.FC<InputBoxProps> = ({ onSendMessage, isLoading }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!isLoading && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isLoading]);

  const submit = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    setInput('');
    await onSendMessage(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-gray-200 p-3 bg-white">
      <div className="flex gap-2 items-end">
        <textarea
          ref={textareaRef}
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Đặt câu hỏi... (Enter để gửi, Shift+Enter để xuống dòng)"
          disabled={isLoading}
          className="flex-1 px-4 py-2.5 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 resize-none text-sm leading-relaxed"
          style={{ maxHeight: '120px', overflowY: 'auto' }}
        />
        <button
          onClick={submit}
          disabled={isLoading || !input.trim()}
          className="px-5 py-2.5 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors text-sm font-medium shrink-0"
        >
          {isLoading ? '...' : 'Gửi'}
        </button>
      </div>
      <p className="text-xs text-gray-400 mt-1.5 ml-1">
        Enter gửi • Shift+Enter xuống dòng
      </p>
    </div>
  );
};
