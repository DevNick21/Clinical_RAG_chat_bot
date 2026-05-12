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
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
