import React from "react";
import { Box, Paper, Typography, Alert } from "@mui/material";
import MessageList from "../components/MessageList";
import ChatInput from "../components/ChatInput";
import useChat from "../hooks/useChat";

const ChatPage = () => {
  const { messages, loading, error, sendMessage, clearChat } = useChat();

  return (
    <Box sx={{ display: "flex", flexDirection: "column", height: "80vh" }}>
      <Typography variant="h4" component="h1" sx={{ mb: 2 }}>
        Clinical RAG Chat
      </Typography>

      <Typography variant="body2" sx={{ mb: 2, color: "text.secondary" }}>
        Ask questions about medical data, patient records, or clinical
        information.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper
        elevation={3}
        sx={{
          p: 2,
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          borderRadius: 2,
          mb: 2,
        }}
      >
        <MessageList messages={messages} loading={loading} />
      </Paper>

      <ChatInput
        onSendMessage={sendMessage}
        onClearChat={clearChat}
        disabled={loading}
      />
    </Box>
  );
};

export default ChatPage;
