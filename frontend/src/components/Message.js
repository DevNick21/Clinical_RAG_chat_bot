/**
 * Message Component - Displays individual chat messages
 */

import React, { useState } from "react";
import {
  Box,
  Paper,
  Typography,
  Avatar,
  Chip,
  IconButton,
  Stack,
  Tooltip,
} from "@mui/material";
import {
  Person as UserIcon,
  SmartToy as BotIcon,
  Error as ErrorIcon,
  ExpandMore as ExpandIcon,
  ContentCopy as CopyIcon,
  CheckCircle as CheckIcon,
  MoreHoriz as StreamingIcon,
} from "@mui/icons-material";

const Message = ({ message }) => {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const isUser = message.type === "user";
  const isBot = message.type === "bot";
  const isError = message.type === "error";
  const isStreaming = message.isStreaming;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const getMessageConfig = () => {
    if (isUser) {
      return {
        align: "flex-end",
        bgcolor: "primary.main",
        color: "white",
        avatar: <UserIcon />,
        avatarBg: "primary.main",
      };
    } else if (isError) {
      return {
        align: "flex-start",
        bgcolor: "error.light",
        color: "white",
        avatar: <ErrorIcon />,
        avatarBg: "error.main",
      };
    } else {
      return {
        align: "flex-start",
        bgcolor: "background.paper",
        color: "text.primary",
        avatar: <BotIcon />,
        avatarBg: "secondary.main",
      };
    }
  };

  const config = getMessageConfig();

  const renderSources = () => {
    if (!message.sources || message.sources.length === 0) return null;

    return (
      <Box sx={{ mt: 2 }}>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ mb: 1, display: "block" }}
        >
          Sources ({message.sources.length}):
        </Typography>
        <Stack spacing={1}>
          {message.sources
            .slice(0, expanded ? undefined : 3)
            .map((source, index) => (
              <Chip
                key={index}
                label={`${source.section || "Unknown"} - ID: ${
                  source.id || "N/A"
                }`}
                size="small"
                variant="outlined"
                color="primary"
              />
            ))}
          {message.sources.length > 3 && (
            <Chip
              label={
                expanded
                  ? "Show less"
                  : `Show ${message.sources.length - 3} more...`
              }
              size="small"
              variant="outlined"
              color="default"
              onClick={() => setExpanded(!expanded)}
              icon={
                <ExpandIcon
                  sx={{ transform: expanded ? "rotate(180deg)" : "none" }}
                />
              }
            />
          )}
        </Stack>
      </Box>
    );
  };

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: config.align,
        alignItems: "flex-start",
        gap: 1,
        mb: 2,
      }}
    >
      {/* Avatar */}
      <Avatar
        sx={{
          bgcolor: config.avatarBg,
          width: 32,
          height: 32,
          order: isUser ? 2 : 0,
        }}
      >
        {config.avatar}
      </Avatar>

      {/* Message Content */}
      <Box sx={{ maxWidth: "75%", minWidth: "200px" }}>
        <Paper
          elevation={isUser ? 2 : 1}
          sx={{
            p: 2,
            bgcolor: config.bgcolor,
            color: config.color,
            border: isError ? "1px solid" : "none",
            borderColor: isError ? "error.main" : "transparent",
            borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
            position: "relative",
          }}
        >
          {/* Copy button */}
          <Tooltip title={copied ? "Copied!" : "Copy message"}>
            <IconButton
              size="small"
              onClick={handleCopy}
              sx={{
                position: "absolute",
                top: 8,
                right: 8,
                opacity: 0.7,
                color: "inherit",
                "&:hover": { opacity: 1 },
              }}
            >
              {copied ? (
                <CheckIcon fontSize="small" />
              ) : (
                <CopyIcon fontSize="small" />
              )}
            </IconButton>
          </Tooltip>

          {/* Message Text */}
          <Typography
            variant="body1"
            sx={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              pr: 5, // Space for copy button
              "& strong": {
                fontWeight: 600,
              },
              "& code": {
                backgroundColor: isUser
                  ? "rgba(255,255,255,0.2)"
                  : "rgba(0,0,0,0.1)",
                padding: "2px 4px",
                borderRadius: "4px",
                fontFamily: "monospace",
                fontSize: "0.9em",
              },
            }}
          >
            {message.content}
            {/* Streaming indicator */}
            {isStreaming && (
              <Box
                component="span"
                sx={{
                  display: "inline-flex",
                  alignItems: "center",
                  ml: 1,
                  animation: "pulse 1.5s ease-in-out infinite",
                  "@keyframes pulse": {
                    "0%, 100%": {
                      opacity: 0.4,
                    },
                    "50%": {
                      opacity: 1,
                    },
                  },
                }}
              >
                <StreamingIcon
                  sx={{
                    fontSize: "1rem",
                    color: "text.secondary",
                  }}
                />
              </Box>
            )}
          </Typography>

          {/* Sources for bot messages */}
          {isBot && renderSources()}
        </Paper>

        {/* Timestamp */}
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            display: "block",
            mt: 0.5,
            textAlign: isUser ? "right" : "left",
            opacity: 0.7,
          }}
        >
          {formatTimestamp(message.timestamp)}
        </Typography>
      </Box>
    </Box>
  );
};

export default Message;
