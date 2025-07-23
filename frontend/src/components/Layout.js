<<<<<<< HEAD
/**
 * Layout Component - Provides consistent layout structure
 */

import React from "react";
import { Box } from "@mui/material";
import Header from "./Header";
import Footer from "./Footer";

const Layout = ({ children }) => {
=======
import React from "react";
import { Outlet } from "react-router-dom";
import { Box, Container } from "@mui/material";
import Header from "./Header";
import Footer from "./Footer";

const Layout = () => {
>>>>>>> 7c90853c1390cb163736bc666c7e2b148c1988b4
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
<<<<<<< HEAD
        backgroundColor: "background.default",
      }}
    >
      <Header />

      <Box
        component="main"
        sx={{ flex: 1, display: "flex", flexDirection: "column" }}
      >
        {children}
      </Box>

=======
        backgroundColor: "#f5f9fd",
      }}
    >
      <Header />
      <Container
        component="main"
        maxWidth="lg"
        sx={{
          mt: 4,
          mb: 4,
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Outlet />
      </Container>
>>>>>>> 7c90853c1390cb163736bc666c7e2b148c1988b4
      <Footer />
    </Box>
  );
};

export default Layout;
