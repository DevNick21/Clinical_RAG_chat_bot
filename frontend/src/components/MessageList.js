/**
 * Message List Component - Displays chat messages
 */

import React, { useEffect, useRef } from "react";
import { Box, Typography, Stack, CircularProgress, Fade } from "@mui/material";
import Message from "./Message";

const MessageList = ({ messages, loading }) => {
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

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
        <Typography variant="body2" sx={{ mt: 3, fontStyle: "italic" }}>
          Try asking: "What diagnoses does patient 10000032 have?" or "Show me
          lab results for admission 25282710"
        </Typography>
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
