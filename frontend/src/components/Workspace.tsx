import React, { useState, useRef, useCallback, useEffect } from 'react';
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
    const [leftWidth, setLeftWidth] = useState(66); // 左侧宽度百分比
    const isDragging = useRef(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const handleMouseDown = () => {
        isDragging.current = true;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    };

    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (!isDragging.current || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const newWidth = ((e.clientX - rect.left) / rect.width) * 100;
        setLeftWidth(Math.min(Math.max(newWidth, 20), 80)); // 限制20%-80%
    }, []);

    const handleMouseUp = useCallback(() => {
        isDragging.current = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    }, []);

    useEffect(() => {
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [handleMouseMove, handleMouseUp]);

    return (
        <div className="workspace">
            <div className="workspace-header">
                <button onClick={onReset} className="back-button">
                    ← Upload New Image
                </button>
            </div>
            <div className="workspace-content" ref={containerRef}>
                <div className="workspace-panel" style={{ width: `${leftWidth}%` }}>
                    <ImageDisplay originalImage={originalImage} resultImage={resultImage} />
                </div>
                <div className="resizer" onMouseDown={handleMouseDown} />
                <div className="workspace-panel" style={{ width: `${100 - leftWidth}%` }}>
                    <ChatInterface
                        messages={messages}
                        onSendMessage={onSendMessage}
                        isProcessing={isProcessing}
                    />
                </div>
            </div>
        </div>
    );
};
