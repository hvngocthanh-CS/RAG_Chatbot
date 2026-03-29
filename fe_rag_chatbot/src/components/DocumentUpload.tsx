import React, { useState } from 'react';
import { chatAPI } from '../services/api';

interface DocumentUploadProps {
  onUploadComplete: (message: string) => void;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({ onUploadComplete }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsLoading(true);
    let successCount = 0;
    let errorCount = 0;

    try {
      // Upload each file individually
      for (const file of Array.from(files)) {
        try {
          const formData = new FormData();
          formData.append('file', file); // Backend expects 'file' not 'files'

          await chatAPI.uploadDocument(formData);
          successCount++;
        } catch (error: any) {
          errorCount++;
          console.error(`Failed to upload ${file.name}:`, error);
        }
      }

      if (successCount > 0) {
        onUploadComplete(`Successfully uploaded ${successCount} document(s)${errorCount > 0 ? `, ${errorCount} failed` : ''}`);
      } else {
        onUploadComplete(`Error: All uploads failed`);
      }
      setIsOpen(false);
    } catch (error: any) {
      onUploadComplete(`Error uploading documents: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
      >
        Upload Documents
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h2 className="text-xl font-bold mb-4">Upload Documents</h2>
            <p className="text-sm text-gray-600 mb-4">
              Supported formats: PDF, DOCX, TXT, Markdown
            </p>

            <label className="block border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-gray-400 transition-colors">
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md"
                onChange={handleFileChange}
                disabled={isLoading}
                className="hidden"
              />
              <div className="text-gray-600">
                {isLoading ? (
                  <p>Uploading...</p>
                ) : (
                  <>
                    <p className="font-medium">Click to select files</p>
                    <p className="text-sm">or drag and drop</p>
                  </>
                )}
              </div>
            </label>

            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setIsOpen(false)}
                disabled={isLoading}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
