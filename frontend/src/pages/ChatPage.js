<<<<<<< HEAD
/**
 * Chat Page Component - Main interface for RAG chat system
 */

import React, { useState, useEffect } from "react";
import {
  Box,
  Paper,
  Typography,
  Alert,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  LinearProgress,
  Stack,
} from "@mui/material";
import {
  Clear as ClearIcon,
  Download as DownloadIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  CheckCircle as ConnectedIcon,
  Error as DisconnectedIcon,
} from "@mui/icons-material";
import { useChat } from "../hooks/useChat";
import MessageList from "../components/MessageList";
import ChatInput from "../components/ChatInput";
import apiService from "../services/api";

const ChatPage = () => {
  const {
    messages,
    loading,
    error,
    isConnected,
    sendMessage,
    clearMessages,
    exportChat,
    getChatStats,
  } = useChat();

  const [models, setModels] = useState(null);
  const [showStats, setShowStats] = useState(false);
  const [showClearDialog, setShowClearDialog] = useState(false);

  // Load available models on component mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const modelData = await apiService.getModels();
        setModels(modelData);
      } catch (error) {
        console.error("Failed to load models:", error);
      }
    };

    loadModels();
  }, []);

  const stats = getChatStats();

  const handleClearConfirm = () => {
    clearMessages();
    setShowClearDialog(false);
  };

  const renderConnectionStatus = () => {
    if (isConnected) {
      return (
        <Chip
          icon={<ConnectedIcon />}
          label="Connected"
          color="success"
          variant="outlined"
          size="small"
        />
      );
    } else {
      return (
        <Chip
          icon={<DisconnectedIcon />}
          label="Disconnected"
          color="error"
          variant="outlined"
          size="small"
        />
      );
    }
  };

  const renderModelInfo = () => {
    if (!models) return null;

    return (
      <Stack direction="row" spacing={1} flexWrap="wrap">
        <Typography variant="caption" color="text.secondary">
          Available Models:
        </Typography>
        {models.embedding_models.slice(0, 3).map((model) => (
          <Chip
            key={model}
            label={model}
            size="small"
            variant="outlined"
            color="primary"
          />
        ))}
        {models.embedding_models.length > 3 && (
          <Chip
            label={`+${models.embedding_models.length - 3} more`}
            size="small"
            variant="outlined"
            color="default"
          />
        )}
      </Stack>
    );
  };

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <Box sx={{ mb: 2 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            mb: 1,
          }}
        >
          <Typography
            variant="h4"
            component="h1"
            sx={{ fontWeight: 600, color: "primary.main" }}
          >
            Clinical RAG Chat
          </Typography>

          <Stack direction="row" spacing={1} alignItems="center">
            {renderConnectionStatus()}

            <Tooltip title="Chat Statistics">
              <IconButton onClick={() => setShowStats(true)}>
                <InfoIcon />
              </IconButton>
            </Tooltip>

            <Tooltip title="Export Chat">
              <IconButton onClick={exportChat} disabled={!stats.hasHistory}>
                <DownloadIcon />
              </IconButton>
            </Tooltip>

            <Tooltip title="Clear Chat">
              <IconButton
                onClick={() => setShowClearDialog(true)}
                disabled={!stats.hasHistory}
              >
                <ClearIcon />
              </IconButton>
            </Tooltip>
          </Stack>
        </Box>

        <Typography variant="body2" color="text.secondary" gutterBottom>
          Ask questions about clinical records from the MIMIC database
        </Typography>

        {renderModelInfo()}

        {/* Connection Warning */}
        {!isConnected && (
          <Alert severity="warning" icon={<WarningIcon />} sx={{ mt: 2 }}>
            Backend server is not responding. Please check if the server is
            running on localhost:5000
          </Alert>
        )}

        {/* Error Display */}
        {error && (
          <Alert
            severity="error"
            sx={{ mt: 2 }}
            onClose={() => {
              /* You might want to add error clearing functionality */
            }}
          >
            {error}
          </Alert>
        )}
      </Box>

      {/* Chat Container */}
      <Paper
        elevation={2}
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          borderRadius: 2,
        }}
      >
        {/* Messages Area */}
        <Box sx={{ flex: 1, overflow: "hidden" }}>
          <MessageList messages={messages} loading={loading} />
        </Box>

        {/* Loading Indicator */}
        {loading && <LinearProgress sx={{ height: 2 }} />}

        {/* Chat Input */}
        <Box sx={{ borderTop: 1, borderColor: "divider" }}>
          <ChatInput
            onSendMessage={sendMessage}
            disabled={loading || !isConnected}
            placeholder={
              !isConnected
                ? "Server disconnected..."
                : loading
                ? "Generating response..."
                : "Ask a question about clinical records..."
            }
          />
        </Box>
      </Paper>

      {/* Statistics Dialog */}
      <Dialog
        open={showStats}
        onClose={() => setShowStats(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Chat Statistics</DialogTitle>
        <DialogContent>
          <Stack spacing={2}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Total Messages
              </Typography>
              <Typography variant="h6">{stats.total}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                User Messages
              </Typography>
              <Typography variant="h6">{stats.user}</Typography>
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Bot Responses
              </Typography>
              <Typography variant="h6">{stats.bot}</Typography>
            </Box>
            {stats.errors > 0 && (
              <Box>
                <Typography variant="body2" color="error.main">
                  Errors
                </Typography>
                <Typography variant="h6" color="error.main">
                  {stats.errors}
                </Typography>
              </Box>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowStats(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Clear Confirmation Dialog */}
      <Dialog open={showClearDialog} onClose={() => setShowClearDialog(false)}>
        <DialogTitle>Clear Chat History</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to clear all messages? This action cannot be
            undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowClearDialog(false)}>Cancel</Button>
          <Button
            onClick={handleClearConfirm}
            color="error"
            variant="contained"
          >
            Clear All
          </Button>
        </DialogActions>
      </Dialog>
=======
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
>>>>>>> 7c90853c1390cb163736bc666c7e2b148c1988b4
    </Box>
  );
};

export default ChatPage;
