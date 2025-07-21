import { useState, useEffect } from "react";
import { sendMessage } from "../services/api";
import {
  saveChatHistory,
  loadChatHistory,
  clearChatHistory,
} from "../services/storage";

/**
 * Custom hook for managing chat functionality
 */
const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load chat history from localStorage on initial load
  useEffect(() => {
    const savedMessages = loadChatHistory();
    if (savedMessages && savedMessages.length > 0) {
      setMessages(savedMessages);
    } else {
      // Set welcome message if no history
      setMessages([
        {
          role: "assistant",
          content:
            "Hello! I'm your Clinical RAG assistant. How can I help you today?",
        },
      ]);
    }
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      saveChatHistory(messages);
    }
  }, [messages]);

  /**
   * Send a message to the API and update messages state
   */
  const handleSendMessage = async (userMessage) => {
    if (!userMessage.trim()) return;

    try {
      // Add user message to chat
      const userMsg = { role: "user", content: userMessage };
      const updatedMessages = [...messages, userMsg];
      setMessages(updatedMessages);
      setLoading(true);
      setError(null);

      // Send to API
      const response = await sendMessage(userMessage, messages);

      // Add assistant response
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: "assistant",
          content: response.response || response.answer || response.message,
        },
      ]);
    } catch (err) {
      console.error("Error sending message:", err);
      setError("Failed to get response. Please try again.");

      // Add error message
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: "assistant",
          content:
            "Sorry, I encountered an error processing your request. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Clear chat history
   */
  const handleClearChat = () => {
    clearChatHistory();
    setMessages([
      {
        role: "assistant",
        content: "Chat history cleared. How can I help you today?",
      },
    ]);
  };

  return {
    messages,
    loading,
    error,
    sendMessage: handleSendMessage,
    clearChat: handleClearChat,
  };
};

export default useChat;
