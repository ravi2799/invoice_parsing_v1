// src/services/api.ts
import axios, { AxiosRequestConfig } from 'axios';
import { PDFService, UploadResponse, OpenAIResponse, APIError } from '../types';

class APIService implements PDFService {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  async uploadPDF(file: File, onProgressUpdate?: (progress: number) => void): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const config: AxiosRequestConfig = {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    };

    if (onProgressUpdate) {
      config.onUploadProgress = (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / (progressEvent.total || 100)
        );
        onProgressUpdate(percentCompleted);
      };
    }

    try {
      const response = await axios.post<UploadResponse>(
        `${this.baseURL}/upload-pdf/`,
        formData,
        config
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error((error.response.data as APIError).detail || 'Error uploading PDF');
      }
      throw new Error('An unexpected error occurred while uploading the PDF');
    }
  }

  async analyzeWithOpenAI(text: string): Promise<OpenAIResponse> {
    try {
      const response = await axios.post<OpenAIResponse>(
        `${this.baseURL}/analyze-with-openai/`,
        { text }
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error((error.response.data as APIError).detail || 'Error analyzing with OpenAI');
      }
      throw new Error('An unexpected error occurred during OpenAI analysis');
    }
  }
}

// Create and export a singleton instance
export const apiService = new APIService('http://localhost:8000');
