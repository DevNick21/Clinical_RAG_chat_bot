import React, { useRef, useEffect } from "react";
import { Box } from "@mui/material";
import Message from "./Message";

const MessageList = ({ messages, loading }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        flexGrow: 1,
        overflowY: "auto",
        px: 2,
        py: 1,
      }}
    >
      {messages.map((message, index) => (
        <Message
          key={index}
          message={message}
          isUser={message.role === "user"}
        />
      ))}

      {loading && (
        <Box sx={{ display: "flex", alignItems: "flex-start", mb: 2 }}>
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </Box>
      )}
      <div ref={messagesEndRef} />
    </Box>
  );
};

export default MessageList;
