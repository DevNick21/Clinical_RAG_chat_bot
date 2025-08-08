/**
 * Chat Input Component - Message input with send functionality
 */

import React, { useState, useRef } from "react";
import {
  Box,
  TextField,
  IconButton,
  Paper,
  Tooltip,
  InputAdornment,
} from "@mui/material";
import { Send as SendIcon, Mic as MicIcon } from "@mui/icons-material";

const ChatInput = ({ onSendMessage, disabled, placeholder }) => {
  const [message, setMessage] = useState("");
  const [isComposing, setIsComposing] = useState(false);
  const textFieldRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!message.trim() || disabled) return;

    onSendMessage(message);
    setMessage("");

    // Focus back to input after sending
    setTimeout(() => {
      textFieldRef.current?.focus();
    }, 100);
  };

  const handleKeyDown = (e) => {
    // Send on Enter (but not Shift+Enter for multi-line)
    if (e.key === "Enter" && !e.shiftKey && !isComposing) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleCompositionStart = () => {
    setIsComposing(true);
  };

  const handleCompositionEnd = () => {
    setIsComposing(false);
  };

  const canSend = message.trim().length > 0 && !disabled;

  return (
    <Box sx={{ p: 2 }}>
      <Paper
        component="form"
        onSubmit={handleSubmit}
        elevation={0}
        sx={{
          display: "flex",
          alignItems: "flex-end",
          gap: 1,
          p: 1,
          backgroundColor: "background.default",
          border: 1,
          borderColor: "divider",
          borderRadius: 3,
          "&:focus-within": {
            borderColor: "primary.main",
            backgroundColor: "background.paper",
          },
          transition: "all 0.2s ease",
        }}
      >
        <TextField
          ref={textFieldRef}
          fullWidth
          multiline
          maxRows={4}
          variant="standard"
          placeholder={
            placeholder || "Ask a question about clinical records..."
          }
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={handleCompositionStart}
          onCompositionEnd={handleCompositionEnd}
          disabled={disabled}
          InputProps={{
            disableUnderline: true,
            endAdornment: (
              <InputAdornment position="end">
                <Box sx={{ display: "flex", gap: 0.5 }}>
                  {/* Voice input button (future enhancement) */}
                  <Tooltip title="Voice input (coming soon)">
                    <span>
                      <IconButton
                        size="small"
                        disabled={true}
                        sx={{ opacity: 0.3 }}
                      >
                        <MicIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Box>
              </InputAdornment>
            ),
          }}
          sx={{
            "& .MuiInputBase-input": {
              padding: "12px 8px",
              fontSize: "1rem",
              lineHeight: 1.5,
              "&::placeholder": {
                color: "text.secondary",
                opacity: 0.7,
              },
            },
          }}
        />

        <Tooltip
          title={canSend ? "Send message (Enter)" : "Type a message to send"}
        >
          <span>
            <IconButton
              type="submit"
              disabled={!canSend}
              color="primary"
              sx={{
                p: 1.5,
                bgcolor: canSend ? "primary.main" : "action.disabled",
                color: canSend ? "white" : "action.disabled",
                borderRadius: 2,
                "&:hover": {
                  bgcolor: canSend ? "primary.dark" : "action.disabled",
                  transform: canSend ? "scale(1.05)" : "none",
                },
                "&:disabled": {
                  bgcolor: "action.disabled",
                  color: "action.disabled",
                },
                transition: "all 0.2s ease",
              }}
            >
              <SendIcon />
            </IconButton>
          </span>
        </Tooltip>
      </Paper>

      {/* Character count for long messages */}
      {message.length > 100 && (
        <Box sx={{ mt: 1, display: "flex", justifyContent: "flex-end" }}>
          <Box
            component="span"
            sx={{
              fontSize: "0.75rem",
              color: message.length > 5000 ? "error.main" : "text.secondary",
              opacity: 0.7,
            }}
          >
            {message.length}/5000 characters
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default ChatInput;
