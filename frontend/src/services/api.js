/**
 * API Service for Clinical RAG Chat
 * Handles all communication with the Flask backend
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

class ApiService {
  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  /**
   * Send a chat message to the RAG system
   * @param {string} message - User message
   * @param {Array} chatHistory - Previous chat messages
   * @returns {Promise<Object>} Response with answer and sources
   */
  async sendMessage(message, chatHistory = []) {
    try {
      const response = await fetch(`${this.baseUrl}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message.trim(),
          chat_history: this.formatChatHistory(chatHistory),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      const data = await response.json();

      // Validate response structure
      if (!data.response) {
        throw new Error("Invalid response format: missing response field");
      }

      return {
        answer: data.response,
        sources: data.sources || [],
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error("API Error:", error);

      // Provide user-friendly error messages
      if (error.message.includes("fetch")) {
        throw new Error(
          "Unable to connect to the server. Please check if the backend is running."
        );
      } else if (error.message.includes("timeout")) {
        throw new Error("Request timed out. The server may be overloaded.");
      } else {
        throw error;
      }
    }
  }

  /**
   * Get available models from the backend
   * @returns {Promise<Object>} Available embedding models and vector stores
   */
  async getModels() {
    try {
      const response = await fetch(`${this.baseUrl}/api/models`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // Validate response structure
      if (!data.embedding_models || !data.vector_stores) {
        throw new Error("Invalid models response format");
      }

      return data;
    } catch (error) {
      console.error("Models API Error:", error);
      throw new Error("Unable to fetch available models");
    }
  }

  /**
   * Format chat history for the API
   * @param {Array} chatHistory - Chat history in frontend format
   * @returns {Array} Chat history in API format
   */
  formatChatHistory(chatHistory) {
    return chatHistory.map((msg) => ({
      role: msg.type === "user" ? "user" : "assistant",
      content: msg.content,
    }));
  }

  /**
   * Check server health
   * @returns {Promise<boolean>} True if server is healthy
   */
  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/api/models`, {
        method: "GET",
        timeout: 5000,
      });
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate message before sending
   * @param {string} message - User message to validate
   * @returns {Object} Validation result
   */
  validateMessage(message) {
    if (!message || typeof message !== "string") {
      return { valid: false, error: "Message must be a non-empty string" };
    }

    const trimmed = message.trim();
    if (trimmed.length === 0) {
      return { valid: false, error: "Message cannot be empty" };
    }

    if (trimmed.length > 5000) {
      return {
        valid: false,
        error: "Message is too long (max 5000 characters)",
      };
    }

    return { valid: true, message: trimmed };
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService;
