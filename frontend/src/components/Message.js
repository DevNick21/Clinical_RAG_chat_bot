import React from "react";
import { Box, Avatar, Typography, Paper } from "@mui/material";
import ReactMarkdown from "react-markdown";
import { SmartToy, Person } from "@mui/icons-material";

const Message = ({ message, isUser }) => {
  return (
    <Box
      sx={{
        display: "flex",
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
    </Box>
  );
};

export default Message;
