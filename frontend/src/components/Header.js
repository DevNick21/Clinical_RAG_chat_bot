import React from "react";
import { AppBar, Toolbar, Typography, Box, Button } from "@mui/material";
import { Link } from "react-router-dom";
import MedicalInformationIcon from "@mui/icons-material/MedicalInformation";

const Header = () => {
  return (
    <AppBar position="static" color="primary" elevation={0}>
      <Toolbar>
        <MedicalInformationIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Clinical RAG Chat
        </Typography>
        <Box>
          <Button color="inherit" component={Link} to="/">
            Chat
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
