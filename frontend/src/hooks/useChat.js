/**
 * Chat Hook for Clinical RAG System
 * Manages chat state, message history, and API communication
 */

import { useState, useCallback, useRef, useEffect } from "react";
import apiService from "../services/api";
import storageService from "../services/storage";

export const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(true);
  const abortControllerRef = useRef(null);

  // Load messages from storage on initialization
  useEffect(() => {
    const savedMessages = storageService.getMessages();
    if (savedMessages && savedMessages.length > 0) {
      setMessages(savedMessages);
    }
  }, []);

  // Save messages to storage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      storageService.saveMessages(messages);
    }
  }, [messages]);

  // Check server health periodically
  useEffect(() => {
    const checkConnection = async () => {
      const healthy = await apiService.checkHealth();
      setIsConnected(healthy);
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  /**
   * Send a message to the RAG system with streaming support
   * @param {string} messageText - The user's message
   */
  const sendMessage = useCallback(
    async (messageText) => {
      // Validate message
      const validation = apiService.validateMessage(messageText);
      if (!validation.valid) {
        setError(validation.error);
        return;
      }

      const userMessage = {
        id: Date.now() + Math.random(),
        type: "user",
        content: validation.message,
        timestamp: new Date().toISOString(),
      };

      // Add user message immediately
      setMessages((prev) => [...prev, userMessage]);
      setLoading(true);
      setError(null);

      // Cancel any previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create a placeholder bot message for streaming updates
      const botMessageId = Date.now() + Math.random() + 1;
      const initialBotMessage = {
        id: botMessageId,
        type: "bot",
        content: "",
        sources: [],
        timestamp: new Date().toISOString(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, initialBotMessage]);

      try {
        // Send streaming message to API
        const response = await apiService.sendMessage(
          validation.message,
          messages.slice(-20), // Send last 20 messages for context
          
          // onChunk callback - updates the bot message in real-time
          (chunk) => {
            setMessages((prev) => 
              prev.map((msg) => 
                msg.id === botMessageId 
                  ? {
                      ...msg, 
                      content: chunk.fullContent,
                      isStreaming: !chunk.done
                    }
                  : msg
              )
            );
          },
          
          // onComplete callback - finalizes the message
          (finalResponse) => {
            setMessages((prev) => 
              prev.map((msg) => 
                msg.id === botMessageId 
                  ? {
                      ...msg,
                      content: finalResponse.answer,
                      sources: finalResponse.sources || [],
                      timestamp: finalResponse.timestamp,
                      metadata: finalResponse.metadata,
                      isStreaming: false
                    }
                  : msg
              )
            );
          }
        );

        console.log("Streaming completed successfully:", response);

      } catch (error) {
        console.error("Chat error:", error);

        // Remove the placeholder bot message and add error message
        setMessages((prev) => {
          const withoutBotMessage = prev.filter((msg) => msg.id !== botMessageId);
          const errorMessage = {
            id: Date.now() + Math.random() + 2,
            type: "error",
            content: `Sorry, I encountered an error: ${error.message}`,
            timestamp: new Date().toISOString(),
          };
          return [...withoutBotMessage, errorMessage];
        });
        
        setError(error.message);
      } finally {
        setLoading(false);
      }
    },
    [messages]
  );

  /**
   * Clear all messages
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    storageService.clearMessages();
  }, []);

  /**
   * Delete a specific message
   * @param {number} messageId - ID of message to delete
   */
  const deleteMessage = useCallback((messageId) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== messageId));
  }, []);

  /**
   * Retry the last failed message
   */
  const retryLastMessage = useCallback(() => {
    if (messages.length >= 2) {
      const lastUserMessage = messages
        .slice()
        .reverse()
        .find((msg) => msg.type === "user");

      if (lastUserMessage) {
        // Remove the last bot/error message and retry
        setMessages((prev) => prev.slice(0, -1));
        sendMessage(lastUserMessage.content);
      }
    }
  }, [messages, sendMessage]);

  /**
   * Export chat history as JSON
   */
  const exportChat = useCallback(() => {
    const exportData = {
      messages: messages,
      exportDate: new Date().toISOString(),
      version: "1.0",
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `clinical-chat-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [messages]);

  /**
   * Get chat statistics
   */
  const getChatStats = useCallback(() => {
    const userMessages = messages.filter((msg) => msg.type === "user");
    const botMessages = messages.filter((msg) => msg.type === "bot");
    const errorMessages = messages.filter((msg) => msg.type === "error");

    return {
      total: messages.length,
      user: userMessages.length,
      bot: botMessages.length,
      errors: errorMessages.length,
      hasHistory: messages.length > 0,
    };
  }, [messages]);

  return {
    messages,
    loading,
    error,
    isConnected,
    sendMessage,
    clearMessages,
    deleteMessage,
    retryLastMessage,
    exportChat,
    getChatStats,
  };
};
