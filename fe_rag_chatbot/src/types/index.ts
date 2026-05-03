export interface Source {
  content: string;
  document_id: string;
  document_name: string;
  page_number?: number;
  chunk_type: string;
  relevance_score: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Source[];
}

export interface ChatRequest {
  question: string;
  conversationId?: string;
  stream?: boolean;
}

export interface DocumentUploadResponse {
  success: boolean;
  message: string;
  documentId?: string;
}

export interface StreamChunk {
  type: 'token' | 'error' | 'done' | 'sources';
  content?: string;
  error?: string;
  conversation_id?: string;
  sources?: Source[];
}
