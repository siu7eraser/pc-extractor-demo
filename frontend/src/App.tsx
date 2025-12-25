import { useState } from 'react';
import { api } from './api';
import './App.css';
import { UploadZone } from './components/UploadZone';
import { Workspace } from './components/Workspace';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

function App() {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = async (file: File) => {
    setIsProcessing(true);
    setError(null);
    try {
      // 1. Create Session
      const { session_id, message } = await api.createSession(file);

      // 2. Set Local State
      setCurrentSessionId(session_id);
      setCurrentImage(URL.createObjectURL(file));
      setMessages([{ id: 'init', role: 'assistant', content: message }]);
    } catch (err: any) {
      setError(err.message || 'Failed to upload image');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSendMessage = async (text: string) => {
    if (!currentSessionId) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setIsProcessing(true);

    try {
      const response = await api.sendMessage(currentSessionId, text);

      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer
      };

      setMessages(prev => [...prev, assistantMsg]);

      if (response.result_image) {
        setResultImage(response.result_image);
      }
    } catch (err: any) {
      console.error(err);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${err.message || 'Something went wrong.'}`
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = async () => {
    if (currentSessionId) {
      try {
        await api.deleteSession(currentSessionId);
      } catch (e) {
        // ignore
      }
    }
    setCurrentSessionId(null);
    setCurrentImage(null);
    setResultImage(null);
    setMessages([]);
    setError(null);
  };

  return (
    <div className="app-container">
      {!currentSessionId ? (
        <div className="upload-screen">
          <h1 className="title">AI Image Segmentation</h1>
          <p className="subtitle">Upload an image and ask to cut out any object.</p>

          <div className="upload-wrapper">
            <UploadZone onFileSelect={handleFileSelect} isProcessing={isProcessing} />
          </div>

          {error && <div className="error-toast">{error}</div>}
        </div>
      ) : (
        <Workspace
          originalImage={currentImage!}
          resultImage={resultImage}
          messages={messages}
          onSendMessage={handleSendMessage}
          isProcessing={isProcessing}
          onReset={handleReset}
        />
      )}
    </div>
  );
}

export default App;
