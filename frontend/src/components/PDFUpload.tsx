// src/components/PDFUpload.tsx
import React, { ChangeEvent, useState } from 'react';

interface PDFUploadProps {
  onFileSelected: (file: File) => void;
  isLoading: boolean;
}

const PDFUpload: React.FC<PDFUploadProps> = ({ onFileSelected, isLoading }) => {
  const [fileName, setFileName] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFileName(selectedFile.name);
      setError(null);
      onFileSelected(selectedFile);
    } else {
      setFileName('');
      setError('Please select a valid PDF file');
    }
  };

  return (
    <div className="pdf-upload">
      <div className="file-input-container">
        <input 
          type="file" 
          onChange={handleFileChange} 
          accept="application/pdf"
          id="file-upload"
          className="file-input"
          disabled={isLoading}
        />
        <label htmlFor="file-upload" className="file-label">
          {fileName || 'Choose PDF file'}
        </label>
      </div>
      
      {error && <p className="error-message">{error}</p>}
    </div>
  );
};

export default PDFUpload;
