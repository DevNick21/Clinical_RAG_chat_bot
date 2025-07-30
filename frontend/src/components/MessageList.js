/**
 * Message List Component - Displays chat messages
 */

import React, { useEffect, useRef, useState } from "react";
import {
  Box,
  Typography,
  Stack,
  CircularProgress,
  Fade,
  Chip,
} from "@mui/material";
import Message from "./Message";
import apiService from "../services/api";

const MessageList = ({ messages, loading, onSuggestionClick }) => {
  const messagesEndRef = useRef(null);
  const [suggestions, setSuggestions] = useState([]);
  const [loadingSuggestions, setLoadingSuggestions] = useState(true);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Load sample suggestions when component mounts
  useEffect(() => {
    const loadSuggestions = async () => {
      try {
        setLoadingSuggestions(true);
        const sampleSuggestions = await apiService.getSampleSuggestions();
        setSuggestions(sampleSuggestions.slice(0, 3)); // Show only first 3
      } catch (error) {
        console.error("Failed to load suggestions:", error);
        // Use fallback suggestions
        setSuggestions([
          "What diagnoses does patient 10000032 have?",
          "Show me lab results for admission 25282710",
          "What medications were prescribed for patient 10006508?",
        ]);
      } finally {
        setLoadingSuggestions(false);
      }
    };

    loadSuggestions();
  }, []);

  const renderWelcomeMessage = () => (
    <Fade in timeout={800}>
      <Box
        sx={{
          textAlign: "center",
          py: 8,
          px: 3,
          color: "text.secondary",
        }}
      >
        <Typography variant="h5" sx={{ mb: 2, fontWeight: 500 }}>
          Welcome to Clinical RAG Assistant
        </Typography>
        <Typography
          variant="body1"
          sx={{ mb: 3, maxWidth: "600px", mx: "auto" }}
        >
          Ask questions about clinical records from the MIMIC database. I can
          help you find information about:
        </Typography>
        <Stack spacing={1} alignItems="center">
          <Typography variant="body2">
            • Patient admissions and demographics
          </Typography>
          <Typography variant="body2">• Diagnoses and ICD codes</Typography>
          <Typography variant="body2">
            • Laboratory results and values
          </Typography>
          <Typography variant="body2">
            • Medications and prescriptions
          </Typography>
          <Typography variant="body2">• Procedures and treatments</Typography>
          <Typography variant="body2">
            • Microbiology culture results
          </Typography>
        </Stack>
        {loadingSuggestions ? (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 3 }}>
            <CircularProgress size={16} />
            <Typography variant="body2" sx={{ fontStyle: "italic" }}>
              Loading sample queries...
            </Typography>
          </Box>
        ) : (
          <Box sx={{ mt: 3 }}>
            <Typography variant="body2" sx={{ mb: 2, fontStyle: "italic" }}>
              Try asking (click to use):
            </Typography>
            <Stack spacing={1} alignItems="center">
              {suggestions.map((suggestion, index) => (
                <Chip
                  key={index}
                  label={`"${suggestion}"`}
                  variant="outlined"
                  size="small"
                  clickable
                  onClick={() =>
                    onSuggestionClick && onSuggestionClick(suggestion)
                  }
                  sx={{
                    maxWidth: "600px",
                    height: "auto",
                    whiteSpace: "normal",
                    textAlign: "center",
                    py: 1,
                    cursor: "pointer",
                    "&:hover": {
                      backgroundColor: "primary.main",
                      borderColor: "primary.main",
                      "& .MuiChip-label": {
                        color: "white",
                      },
                    },
                    "& .MuiChip-label": {
                      display: "block",
                      whiteSpace: "normal",
                      lineHeight: 1.2,
                    },
                  }}
                />
              ))}
            </Stack>
          </Box>
        )}
      </Box>
    </Fade>
  );

  const renderLoadingIndicator = () => (
    <Fade in timeout={300}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 2,
          p: 2,
          justifyContent: "flex-start",
          ml: 2,
        }}
      >
        <CircularProgress size={20} />
        <Typography variant="body2" color="text.secondary">
          Analyzing clinical records...
        </Typography>
      </Box>
    </Fade>
  );

  return (
    <Box
      sx={{
        height: "100%",
        overflow: "auto",
        display: "flex",
        flexDirection: "column",
        "&::-webkit-scrollbar": {
          width: "8px",
        },
        "&::-webkit-scrollbar-track": {
          backgroundColor: "transparent",
        },
        "&::-webkit-scrollbar-thumb": {
          backgroundColor: "rgba(0,0,0,0.2)",
          borderRadius: "4px",
          "&:hover": {
            backgroundColor: "rgba(0,0,0,0.3)",
          },
        },
      }}
    >
      {messages.length === 0 ? (
        renderWelcomeMessage()
      ) : (
        <Box sx={{ flex: 1, p: 2 }}>
          <Stack spacing={2}>
            {messages.map((message, index) => (
              <Fade
                key={message.id}
                in
                timeout={500}
                style={{ transitionDelay: `${index * 100}ms` }}
              >
                <div>
                  <Message message={message} />
                </div>
              </Fade>
            ))}
          </Stack>

          {/* Loading indicator */}
          {loading && renderLoadingIndicator()}

          {/* Auto-scroll anchor */}
          <div ref={messagesEndRef} />
        </Box>
      )}
    </Box>
  );
};

export default MessageList;
