import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  IconButton, 
  Paper
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';

const ChatInput = ({ onSendMessage, onClearChat, disabled }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <Paper 
      elevation={2}
      component="form"
      onSubmit={handleSubmit}
      sx={{ 
        p: 2, 
        display: 'flex',
        alignItems: 'center',
        borderRadius: 2,
      }}
    >
      <TextField
        fullWidth
        placeholder="Ask a medical question..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        variant="outlined"
        disabled={disabled}
        autoComplete="off"
        sx={{ mr: 1 }}
      />
      <Box sx={{ display: 'flex' }}>
        <IconButton 
          color="error" 
          onClick={onClearChat}
          disabled={disabled}
          size="large"
        >
          <DeleteIcon />
        </IconButton>
        <IconButton 
          color="primary" 
          type="submit" 
          disabled={!message.trim() || disabled}
          size="large"
        >
          <SendIcon />
        </IconButton>
      </Box>
    </Paper>
  );
};

export default ChatInput;
