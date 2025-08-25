/**
 * Footer Component
 */

import React from "react";
import { Box, Typography, Container } from "@mui/material";

const Footer = () => {
  return (
    <Box
      component="footer"
      sx={{
        py: 2,
        px: 2,
        mt: "auto",
        backgroundColor: "background.paper",
        borderTop: 1,
        borderColor: "divider",
      }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 1,
          }}
        >
          <Typography variant="body2" color="text.secondary">
            © 2025 Clinical RAG Assistant
          </Typography>

          <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
            <Typography variant="caption" color="text.secondary">
              MIMIC Database Integration
            </Typography>
            <Typography variant="caption" color="text.secondary">
              By Ekenedirichukwu Iheanacho
            </Typography>
          </Box>
        </Box>

        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            display: "block",
            textAlign: "center",
            mt: 1,
            fontStyle: "italic",
          }}
        >
           This system is for educational purposes only.
        </Typography>
      </Container>
    </Box>
  );
};

export default Footer;
