import React from "react";
import { Box, Typography, Link } from "@mui/material";

const Footer = () => {
  return (
    <Box
      component="footer"
      sx={{
        py: 2,
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
    </Box>
  );
};

export default Footer;
