import axios from "axios";

// Base URL for API calls
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Send a chat message to the Clinical RAG API
 * @param {string} message - User's message
 * @param {Array} chatHistory - Previous chat history
 * @returns {Promise} - API response
 */
export const sendMessage = async (message, chatHistory = []) => {
  try {
    const response = await api.post("/chat", {
      message,
      chat_history: chatHistory.map((msg) => ({
        role: msg.role,
        content: msg.content,
      })),
    });
    return response.data;
  } catch (error) {
    console.error("Error sending message:", error);
    throw error;
  }
};

/**
 * Fetch available embedding models
 * @returns {Promise} - API response with available models
 */
export const getAvailableModels = async () => {
  try {
    const response = await api.get("/models");
    return response.data;
  } catch (error) {
    console.error("Error fetching models:", error);
    throw error;
  }
};

export default {
  sendMessage,
  getAvailableModels,
};
