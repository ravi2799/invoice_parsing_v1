


import React, { useState, ChangeEvent } from 'react';
import axios from 'axios';
import './App.css';

interface PageResponse {
  page: number;
  text: string;
  error?: string;
}

interface UploadResponse {
  filename: string;
  pages: PageResponse[];
  errors: PageResponse[];
  final_summary: string;
}

type DocumentType = 'invoice' | 'freight' | 'customs';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadType, setUploadType] = useState<DocumentType>('invoice');
  const [activeTab, setActiveTab] = useState<'upload' | 'results'>('upload');

  const API_URL = 'http://localhost:8000';

  const documentTypeLabels = {
    invoice: 'Commercial Invoice',
    freight: 'Freight Document',
    customs: 'Customs Declaration'
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];

    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError(null);
      setUploadResponse(null);
    } else {
      setFile(null);
      setFileName('');
      setError('Please select a valid PDF file');
    }
  };

  const getEndpointForDocumentType = (type: DocumentType): string => {
    switch(type) {
      case 'invoice':
        return `${API_URL}/upload-commercial-invoice/`;
      case 'freight':
        return `${API_URL}/upload-fret/`;
      case 'customs':
        return `${API_URL}/upload-custom-declaration/`;
      default:
        return `${API_URL}/upload-commercial-invoice/`;
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Select the appropriate endpoint based on upload type
      const endpoint = getEndpointForDocumentType(uploadType);

      const response = await axios.post<UploadResponse>(
        endpoint,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / (progressEvent.total || 100)
            );
            setUploadProgress(percentCompleted);
          },
        }
      );

      setUploadResponse(response.data);
      setUploadProgress(100);
      setActiveTab('results');
    } catch (err) {
      console.error('Error uploading file:', err);
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail || 'Error uploading file');
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>PDF Parser & Analyzer</h1>
        <p>Upload a PDF document and analyze it with OpenAI</p>
      </header>

      <main>
        <div className="tab-navigation">
          <button 
            className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            Upload
          </button>
          {uploadResponse && (
            <button 
              className={`tab-button ${activeTab === 'results' ? 'active' : ''}`}
              onClick={() => setActiveTab('results')}
            >
              Results
            </button>
          )}
        </div>

        {activeTab === 'upload' && (
          <section className="upload-section">
            <div className="document-type-selector">
              <h2>Select Document Type</h2>
              <div className="type-buttons">
                <button 
                  className={`type-button ${uploadType === 'invoice' ? 'active' : ''}`}
                  onClick={() => setUploadType('invoice')}
                >
                  Commercial Invoice
                </button>
                <button 
                  className={`type-button ${uploadType === 'freight' ? 'active' : ''}`}
                  onClick={() => setUploadType('freight')}
                >
                  Freight Document
                </button>
                <button 
                  className={`type-button ${uploadType === 'customs' ? 'active' : ''}`}
                  onClick={() => setUploadType('customs')}
                >
                  Customs Declaration
                </button>
              </div>
            </div>
            
            <h2>Upload {documentTypeLabels[uploadType]} PDF</h2>
            <div className="file-input-container">
              <input
                type="file"
                onChange={handleFileChange}
                accept="application/pdf"
                id="file-upload"
                className="file-input"
              />
              <label htmlFor="file-upload" className="file-label">
                {fileName || `Choose ${documentTypeLabels[uploadType]} PDF file`}
              </label>
              <button
                onClick={handleUpload}
                disabled={!file || isLoading}
                className="upload-button"
              >
                Upload & Analyze
              </button>
            </div>

            {uploadProgress > 0 && uploadProgress < 100 && (
              <div className="progress-container">
                <div
                  className="progress-bar"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
                <span>{uploadProgress}%</span>
              </div>
            )}

            {error && <p className="error-message">{error}</p>}
          </section>
        )}

        {activeTab === 'results' && uploadResponse && (
          <div className="results-container">
            <section className="results-header">
              <h2>Analysis Results</h2>
              <h3>Document: {uploadResponse.filename}</h3>
              <p>Type: {documentTypeLabels[uploadType]}</p>
            </section>

            {uploadResponse.errors.length > 0 && (
              <section className="errors-section">
                <h3>Processing Errors</h3>
                <div className="errors-list">
                  {uploadResponse.errors.map((err) => (
                    <div key={err.page} className="error-item">
                      <span className="error-page">Page {err.page}:</span> {err.error}
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="summary-section">
              <h3>Document Summary</h3>
              <div className="summary-content">
                <pre className="final-summary">{JSON.stringify(uploadResponse.final_summary, null, 2)}</pre>
              </div>
            </section>

            <section className="raw-data-section">
              <details>
                <summary>View Raw Page Analysis</summary>
                <div className="pages-container">
                  {uploadResponse.pages.map((page) => (
                    <div key={page.page} className="page-analysis">
                      <h4>Page {page.page}</h4>
                      <pre className="page-text">{page.text}</pre>
                    </div>
                  ))}
                </div>
              </details>
            </section>
          </div>
        )}

        {isLoading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <p>Processing document. This may take a moment...</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;