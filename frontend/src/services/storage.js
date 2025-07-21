/**
 * Local storage service for persisting chat history
 */

// Key for storing chat history in localStorage
const CHAT_HISTORY_KEY = "clinical_rag_chat_history";

/**
 * Save chat history to local storage
 * @param {Array} chatHistory - Array of message objects
 */
export const saveChatHistory = (chatHistory) => {
  try {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistory));
  } catch (error) {
    console.error("Error saving chat history:", error);
  }
};

/**
 * Load chat history from local storage
 * @returns {Array} - Array of message objects or empty array if not found
 */
export const loadChatHistory = () => {
  try {
    const savedHistory = localStorage.getItem(CHAT_HISTORY_KEY);
    return savedHistory ? JSON.parse(savedHistory) : [];
  } catch (error) {
    console.error("Error loading chat history:", error);
    return [];
  }
};

/**
 * Clear chat history from local storage
 */
export const clearChatHistory = () => {
  try {
    localStorage.removeItem(CHAT_HISTORY_KEY);
  } catch (error) {
    console.error("Error clearing chat history:", error);
  }
};

export default {
  saveChatHistory,
  loadChatHistory,
  clearChatHistory,
};
