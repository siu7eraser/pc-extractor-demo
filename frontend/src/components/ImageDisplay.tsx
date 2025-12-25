import React from 'react';
import './ImageDisplay.css';

interface ImageDisplayProps {
    originalImage: string; // URL.createObjectURL or Base64
    resultImage: string | null; // Base64
}

export const ImageDisplay: React.FC<ImageDisplayProps> = ({ originalImage, resultImage }) => {
    return (
        <div className="image-display">
            <div className="image-wrapper">
                {/* Original Image Layer */}
                <img
                    src={originalImage}
                    alt="Original"
                    className="original-layer"
                />

                {/* Result Overlay Layer - Smooth transition when available */}
                {resultImage && (
                    <div className="result-layer-container">
                        <img
                            src={resultImage}
                            alt="Segmented Result"
                            className="result-layer"
                        />
                    </div>
                )}
            </div>

            {resultImage && (
                <div className="image-controls">
                    <span className="badge">Segmentation Result</span>
                </div>
            )}
        </div>
    );
};
