/**
 * Error Boundary Component - Catches and displays React errors gracefully
 */

import React from "react";
import { Box, Typography, Button, Container, Alert } from "@mui/material";
import { Refresh as RefreshIcon } from "@mui/icons-material";

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to console and potentially to error reporting service
    console.error("Error caught by boundary:", error, errorInfo);
    this.setState({
      error: error,
      errorInfo: errorInfo,
    });
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <Container maxWidth="md" sx={{ mt: 8, mb: 4 }}>
          <Box
            sx={{
              textAlign: "center",
              p: 4,
              bgcolor: "background.paper",
              borderRadius: 2,
              boxShadow: 1,
            }}
          >
            <Typography variant="h4" color="error" gutterBottom>
              Oops! Something went wrong
            </Typography>

            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              The Clinical RAG Assistant encountered an unexpected error. This
              might be a temporary issue.
            </Typography>

            <Alert severity="error" sx={{ mb: 3, textAlign: "left" }}>
              <Typography variant="body2">
                <strong>Error:</strong>{" "}
                {this.state.error && this.state.error.toString()}
              </Typography>
              {process.env.NODE_ENV === "development" &&
                this.state.errorInfo && (
                  <Typography
                    variant="caption"
                    component="pre"
                    sx={{ mt: 2, overflow: "auto" }}
                  >
                    {this.state.errorInfo.componentStack}
                  </Typography>
                )}
            </Alert>

            <Box
              sx={{
                display: "flex",
                gap: 2,
                justifyContent: "center",
                flexWrap: "wrap",
              }}
            >
              <Button
                variant="contained"
                startIcon={<RefreshIcon />}
                onClick={this.handleReload}
                color="primary"
              >
                Reload Page
              </Button>

              <Button
                variant="outlined"
                onClick={this.handleReset}
                color="secondary"
              >
                Try Again
              </Button>
            </Box>

            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ mt: 3, display: "block" }}
            >
              If this problem persists, please check the browser console for
              more details.
            </Typography>
          </Box>
        </Container>
      );
    }

    return this.props.children;
  }
}
