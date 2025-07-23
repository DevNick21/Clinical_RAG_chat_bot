<<<<<<< HEAD
/**
 * Header Component
 */

import React from "react";
import { AppBar, Toolbar, Typography, Box } from "@mui/material";
import { LocalHospital as MedicalIcon } from "@mui/icons-material";

const Header = () => {
  return (
    <AppBar position="static" elevation={0} sx={{ bgcolor: "primary.main" }}>
      <Toolbar>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <MedicalIcon sx={{ fontSize: 28 }} />
          <Typography
            variant="h6"
            component="div"
            sx={{
              fontWeight: 600,
              letterSpacing: "-0.5px",
            }}
          >
            Clinical RAG Assistant
          </Typography>
        </Box>

        <Box sx={{ ml: "auto" }}>
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            MIMIC Database Query System
          </Typography>
=======
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
>>>>>>> 7c90853c1390cb163736bc666c7e2b148c1988b4
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
