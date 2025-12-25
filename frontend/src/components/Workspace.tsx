import React from 'react';
import { ImageDisplay } from './ImageDisplay';
import { ChatInterface } from './ChatInterface';
import './Workspace.css';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
}

interface WorkspaceProps {
    originalImage: string;
    resultImage: string | null;
    messages: Message[];
    onSendMessage: (message: string) => Promise<void>;
    isProcessing: boolean;
    onReset: () => void;
}

export const Workspace: React.FC<WorkspaceProps> = ({
    originalImage,
    resultImage,
    messages,
    onSendMessage,
    isProcessing,
    onReset
}) => {
    return (
        <div className="workspace">
            <div className="workspace-header">
                <button onClick={onReset} className="back-button">
                    ← Upload New Image
                </button>
            </div>
            <div className="workspace-content">
                <ImageDisplay originalImage={originalImage} resultImage={resultImage} />
                <ChatInterface
                    messages={messages}
                    onSendMessage={onSendMessage}
                    isProcessing={isProcessing}
                />
            </div>
        </div>
    );
};
