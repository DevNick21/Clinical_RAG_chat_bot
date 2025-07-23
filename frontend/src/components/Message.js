<<<<<<< HEAD
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
} from "@mui/icons-material";

const Message = ({ message }) => {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const isUser = message.type === "user";
  const isBot = message.type === "bot";
  const isError = message.type === "error";

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

=======
import React from "react";
import { Box, Avatar, Typography, Paper } from "@mui/material";
import ReactMarkdown from "react-markdown";
import { SmartToy, Person } from "@mui/icons-material";

const Message = ({ message, isUser }) => {
>>>>>>> 7c90853c1390cb163736bc666c7e2b148c1988b4
  return (
    <Box
      sx={{
        display: "flex",
<<<<<<< HEAD
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
=======
        alignItems: "flex-start",
        mb: 2,
        flexDirection: isUser ? "row-reverse" : "row",
      }}
    >
      <Avatar
        sx={{
          bgcolor: isUser ? "primary.main" : "secondary.main",
          mr: isUser ? 0 : 1,
          ml: isUser ? 1 : 0,
        }}
      >
        {isUser ? <Person /> : <SmartToy />}
      </Avatar>
      <Paper
        elevation={1}
        sx={{
          p: 2,
          maxWidth: "70%",
          borderRadius: 2,
          backgroundColor: isUser ? "primary.light" : "background.paper",
          color: isUser ? "white" : "text.primary",
          "& p": {
            marginTop: 0,
            marginBottom: 0,
          },
        }}
      >
        {isUser ? (
          <Typography>{message.content}</Typography>
        ) : (
          <ReactMarkdown>{message.content}</ReactMarkdown>
        )}
      </Paper>
>>>>>>> 7c90853c1390cb163736bc666c7e2b148c1988b4
    </Box>
  );
};

export default Message;
