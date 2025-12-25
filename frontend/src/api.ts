
export const API_BASE_URL = 'http://localhost:5000';

export interface CreateSessionResponse {
  session_id: string;
  message: string;
}

export interface ChatResponse {
  answer: string;
  result_image: string | null;
  session_id: string;
}

export interface ErrorResponse {
  error: string;
}

export const api = {
  /**
   * Create a new session with an image
   */
  createSession: async (imageFile: File): Promise<CreateSessionResponse> => {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/api/session/create`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || 'Failed to create session');
    }

    return response.json();
  },

  /**
   * Send a message to the session
   */
  sendMessage: async (sessionId: string, message: string): Promise<ChatResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/session/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        message: message,
      }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || 'Failed to send message');
    }

    return response.json();
  },

  /**
   * Delete a session
   */
  deleteSession: async (sessionId: string): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/api/session/delete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
      }),
    });
    
    // We don't strictly care if delete fails for the UI flow, but good to know
    if (!response.ok) {
       console.warn('Failed to delete session');
    }
  }
};
