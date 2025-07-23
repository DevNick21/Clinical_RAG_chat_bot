<<<<<<< HEAD
/**
 * Footer Component
 */

import React from "react";
import { Box, Typography, Container } from "@mui/material";
=======
import React from "react";
import { Box, Typography, Link } from "@mui/material";
>>>>>>> 7c90853c1390cb163736bc666c7e2b148c1988b4

const Footer = () => {
  return (
    <Box
      component="footer"
      sx={{
        py: 2,
<<<<<<< HEAD
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
            © 2025 Clinical RAG Assistant - Educational Purpose Only
          </Typography>

          <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
            <Typography variant="caption" color="text.secondary">
              MIMIC Database Integration
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Powered by LangChain & FAISS
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
          ⚠️ This system is for educational purposes only. Always consult
          qualified healthcare professionals for medical advice.
        </Typography>
      </Container>
=======
        textAlign: "center",
        backgroundColor: "primary.main",
        color: "white",
      }}
    >
      <Typography variant="body2">
        © {new Date().getFullYear()} Clinical RAG Chat | Built with LangChain &
        React
      </Typography>
      <Typography variant="caption">
        <Link color="inherit" href="https://github.com/DevNick21/msc_project">
          Github Repository
        </Link>
      </Typography>
>>>>>>> 7c90853c1390cb163736bc666c7e2b148c1988b4
    </Box>
  );
};

export default Footer;
