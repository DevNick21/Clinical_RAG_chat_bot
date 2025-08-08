/**
 * Storage Service for Clinical RAG Chat
 * Handles local storage operations for chat persistence
 */

const STORAGE_KEYS = {
  MESSAGES: "clinical_rag_messages",
  PREFERENCES: "clinical_rag_preferences",
  SESSION: "clinical_rag_session",
};

class StorageService {
  constructor() {
    this.isSupported = this.checkSupport();
  }

  /**
   * Check if localStorage is supported
   * @returns {boolean} True if localStorage is available
   */
  checkSupport() {
    try {
      const test = "localStorage-test";
      localStorage.setItem(test, test);
      localStorage.removeItem(test);
      return true;
    } catch (e) {
      console.warn("localStorage not available:", e);
      return false;
    }
  }

  /**
   * Save messages to localStorage
   * @param {Array} messages - Chat messages to save
   */
  saveMessages(messages) {
    if (!this.isSupported) return;

    try {
      const data = {
        messages: messages,
        timestamp: new Date().toISOString(),
        version: "1.0",
      };
      localStorage.setItem(STORAGE_KEYS.MESSAGES, JSON.stringify(data));
    } catch (error) {
      console.error("Error saving messages:", error);
    }
  }

  /**
   * Get messages from localStorage
   * @returns {Array} Saved chat messages
   */
  getMessages() {
    if (!this.isSupported) return [];

    try {
      const data = localStorage.getItem(STORAGE_KEYS.MESSAGES);
      if (!data) return [];

      const parsed = JSON.parse(data);
      return parsed.messages || [];
    } catch (error) {
      console.error("Error loading messages:", error);
      return [];
    }
  }

  /**
   * Clear all saved messages
   */
  clearMessages() {
    if (!this.isSupported) return;

    try {
      localStorage.removeItem(STORAGE_KEYS.MESSAGES);
    } catch (error) {
      console.error("Error clearing messages:", error);
    }
  }

  /**
   * Save user preferences
   * @param {Object} preferences - User preferences to save
   */
  savePreferences(preferences) {
    if (!this.isSupported) return;

    try {
      localStorage.setItem(
        STORAGE_KEYS.PREFERENCES,
        JSON.stringify(preferences)
      );
    } catch (error) {
      console.error("Error saving preferences:", error);
    }
  }

  /**
   * Get user preferences
   * @returns {Object} Saved user preferences
   */
  getPreferences() {
    if (!this.isSupported) return {};

    try {
      const data = localStorage.getItem(STORAGE_KEYS.PREFERENCES);
      return data ? JSON.parse(data) : {};
    } catch (error) {
      console.error("Error loading preferences:", error);
      return {};
    }
  }

  /**
   * Save session data
   * @param {Object} sessionData - Session data to save
   */
  saveSession(sessionData) {
    if (!this.isSupported) return;

    try {
      const data = {
        ...sessionData,
        timestamp: new Date().toISOString(),
      };
      sessionStorage.setItem(STORAGE_KEYS.SESSION, JSON.stringify(data));
    } catch (error) {
      console.error("Error saving session:", error);
    }
  }

  /**
   * Get session data
   * @returns {Object} Saved session data
   */
  getSession() {
    try {
      const data = sessionStorage.getItem(STORAGE_KEYS.SESSION);
      return data ? JSON.parse(data) : {};
    } catch (error) {
      console.error("Error loading session:", error);
      return {};
    }
  }

  /**
   * Clear session data
   */
  clearSession() {
    try {
      sessionStorage.removeItem(STORAGE_KEYS.SESSION);
    } catch (error) {
      console.error("Error clearing session:", error);
    }
  }

  /**
   * Get storage usage statistics
   * @returns {Object} Storage usage information
   */
  getStorageStats() {
    if (!this.isSupported) {
      return { supported: false };
    }

    try {
      const messages = localStorage.getItem(STORAGE_KEYS.MESSAGES);
      const preferences = localStorage.getItem(STORAGE_KEYS.PREFERENCES);
      const session = sessionStorage.getItem(STORAGE_KEYS.SESSION);

      return {
        supported: true,
        messagesSize: messages ? messages.length : 0,
        preferencesSize: preferences ? preferences.length : 0,
        sessionSize: session ? session.length : 0,
        totalLocalStorage: JSON.stringify(localStorage).length,
        totalSessionStorage: JSON.stringify(sessionStorage).length,
      };
    } catch (error) {
      console.error("Error getting storage stats:", error);
      return { supported: true, error: error.message };
    }
  }

  /**
   * Clear all stored data
   */
  clearAll() {
    this.clearMessages();
    this.clearSession();

    if (this.isSupported) {
      try {
        localStorage.removeItem(STORAGE_KEYS.PREFERENCES);
      } catch (error) {
        console.error("Error clearing preferences:", error);
      }
    }
  }
}

// Create singleton instance
const storageService = new StorageService();

export default storageService;
