import axios from 'axios';
import { ChatRequest, StreamChunk } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatAPI = {
  async* streamChat(request: ChatRequest): AsyncGenerator<StreamChunk> {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: request.question,
          conversation_id: request.conversationId,
          stream: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      
      if (!reader) {
        throw new Error('Stream not supported');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          yield { type: 'done' };
          break;
        }

        // Accumulate chunks; {stream:true} handles multi-byte char boundaries
        buffer += decoder.decode(value, { stream: true });

        // Split on newlines but keep the last incomplete line in the buffer
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'sources') {
              yield {
                type: 'sources' as const,
                conversation_id: data.conversation_id,
              };
            } else if (data.type === 'done') {
              yield { type: 'done' };
            } else if (data.type === 'error') {
              yield { type: 'error', error: data.error || 'Unknown error' };
            } else {
              yield {
                type: 'token' as const,
                content: data.token || data.content || '',
              };
            }
          } catch {
            // Incomplete or invalid JSON — wait for next chunk
          }
        }
      }
    } catch (error: any) {
      yield {
        type: 'error',
        error: error.message || 'Failed to stream chat',
      };
    }
  },

  async uploadDocument(formData: FormData): Promise<any> {
    try {
      const response = await apiClient.post('/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to upload document');
    }
  },

  async getHealth(): Promise<any> {
    try {
      const response = await fetch('http://localhost:8000/health/live');
      return await response.json();
    } catch (error: any) {
      throw new Error('Health check failed');
    }
  },
};
