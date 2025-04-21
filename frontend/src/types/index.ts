// src/types/index.ts

// API response types
export interface UploadResponse {
  filename: string;
  text: string;
}

export interface OpenAIResponse {
  text: string;
}

export interface APIError {
  detail: string;
}

// Service types
export interface PDFService {
  uploadPDF(file: File, onProgressUpdate?: (progress: number) => void): Promise<UploadResponse>;
  analyzeWithOpenAI(text: string): Promise<OpenAIResponse>;
}
