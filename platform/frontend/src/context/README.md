# React Context Folder

This directory is intended for React Context API files that provide state management across the application.

## What is React Context?

React Context is a way to share state between components without having to explicitly pass props down through every level of the component tree. This is useful for data that can be considered "global" for a tree of React components.

## Potential Uses in This Application

Future context providers could include:

- ChatContext - Manage chat history and state
- UserPreferencesContext - Store user preferences like theme
- ModelSelectionContext - Handle model selection for the RAG system
- AuthContext - Manage authentication state

## How to Use

To create a new context, add a new file like `ChatContext.js` with the following structure:

```javascript
import React, { createContext, useState, useContext } from 'react';

const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
  const [chatHistory, setChatHistory] = useState([]);

  const addMessage = (message) => {
    setChatHistory([...chatHistory, message]);
  };

  return (
    <ChatContext.Provider value={{ chatHistory, addMessage }}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => useContext(ChatContext);
```

Then wrap your application or component tree in the provider:

```javascript
// In App.js or another high-level component
import { ChatProvider } from './context/ChatContext';

function App() {
  return (
    <ChatProvider>
      {/* Your app components */}
    </ChatProvider>
  );
}
```
