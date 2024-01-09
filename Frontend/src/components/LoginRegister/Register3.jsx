import React, { useState } from "react";
// import { Link, useHistory } from "react-router-dom";
import NavbarLogin from "./NavbarLogin";
import "../../App.css";
import {
  CssBaseline,
  TextField,
  Button,
  Typography,
  Paper,
  Box,
  Link,
  Container,
} from "@mui/material";
import circularLogo from "../../images/circularLogo.png";
import { createTheme, ThemeProvider } from "@mui/material/styles";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import Visibility from "@mui/icons-material/Visibility";
import VisibilityOff from "@mui/icons-material/VisibilityOff";

import VerificationModal from "./VerificationModal";
import { CognitoUser } from "amazon-cognito-identity-js";
import UserPool from "../../UserPool";

export default function Register3() {
  const theme = createTheme({
    palette: {
      custom: {
        main: "#3B533A",
      },
    },
  });

  const [showPassword, setShowPassword] = React.useState(false);

  const handleClickShowPassword = () => setShowPassword((show) => !show);

  const handleMouseDownPassword = (event) => {
    event.preventDefault();
  };

  const [isVerificationModalOpen, setIsVerificationModalOpen] = useState(false);

  const [isVerificationModalClose, setIsVerificationModalClose] =
    useState(false);

  const openVerificationModal = () => {
    setIsVerificationModalOpen(true);
  };

  const closeVerificationModal = () => {
    setIsVerificationModalOpen(false);
  };

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [user, setUser] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);


  // const onSubmit = (event) => {
  //   event.preventDefault();

  //   if (!email || !password) {
  //     alert("Please enter both email and password.");
  //     return;
  //   }

  //   UserPool.signUp(email, password, [], null, (err, data) => {
  //     if (err) {
  //       if (err.code === "UsernameExistsException") {
  //         alert("User already exists. Please verify your email.");
  //         openVerificationModal(); // Reopen the modal for verification
  //       } else if (err.code === "LimitExceededException") {
  //         alert("Exceeded daily email limit for the operation or the account. Please try again later or contact support for assistance.");
  //       } else {
  //         console.error(err);
  //         alert("An error occurred during registration. Please try again.");
  //       }
  //       return;
  //     }

  //     const cognitoUser = new CognitoUser({
  //       Username: data.user.getUsername(),
  //       Pool: UserPool,
  //     });

  //     setUser(cognitoUser);
  //     openVerificationModal();
  //   });
  // };

  const validatePassword = (password) => {
    const minLength = 8;
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumber = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
  
    return (
      password.length >= minLength &&
      hasUpperCase &&
      hasLowerCase &&
      hasNumber &&
      hasSpecialChar
    );
  };

  const onSubmit = (event) => {
    event.preventDefault();

    // Prevent multiple submissions
    if (isSubmitting) {
        return;
    }
    setIsSubmitting(true); // Start of submission process

    // Check if email and password are provided
    if (!email || !password) {
      alert("Please enter both email and password.");
      setIsSubmitting(false); // Reset submission status
      return;
    }

    if (!validatePassword(password)) {
      alert("Password must be at least 8 characters long and include at least one uppercase letter, one lowercase letter, one number, and one special character.");
      setIsSubmitting(false);
      return;
    }

    // Proceed with UserPool signUp
    UserPool.signUp(email, password, [], null, (err, data) => {
      if (err) {
        if (err.code === "UsernameExistsException") {
          alert("User already exists. Please verify your email.");
          openVerificationModal(); // Reopen the modal for verification
        } else if (err.code === "LimitExceededException") {
          alert("Exceeded daily email limit for the operation or the account. Please try again later or contact support for assistance.");
        } else {
          console.error(err);
          alert("An error occurred during registration. Please try again.");
        }
        setIsSubmitting(false); // Reset submission status
        return;
      }

      // Successful registration
      const cognitoUser = new CognitoUser({
        Username: data.user.getUsername(),
        Pool: UserPool,
      });

      setUser(cognitoUser);
      openVerificationModal();

      setIsSubmitting(false); // Reset submission status
    });
};



  return (
    <div>
      <NavbarLogin />
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          style={{
            width: "100vw",
            height: "100vh",
            backgroundColor: "#9FA59A",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Container component="main" maxWidth="xs">
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <Paper
                elevation={3}
                style={{
                  padding: "20px",
                  width: "100%",
                  position: "relative",
                  overflow: "visible",
                  borderRadius: "16px",
                  marginTop: "60px",
                  marginBottom: "60px",
                }}
              >
                <Box
                  style={{
                    position: "absolute",
                    top: "-40px",
                    left: "50%",
                    transform: "translateX(-50%)",
                    width: "80px",
                    height: "80px",
                  }}
                >
                  <img
                    src={circularLogo}
                    alt="logo"
                    style={{ width: "100%", height: "100%" }}
                  />
                </Box>
                <Typography
                  component="h1"
                  variant="h5"
                  style={{
                    textAlign: "center",
                    marginTop: "60px",
                    fontWeight: "bold",
                    fontSize: "20px",
                  }}
                >
                  Create account
                </Typography>
                <form
                  onSubmit={onSubmit}
                  style={{
                    width: "100%",
                    marginTop: "20px",
                  }}
                  noValidate
                >
                  <TextField
                    variant="outlined"
                    margin="normal"
                    required
                    fullWidth
                    id="email"
                    label="Email Address"
                    name="email"
                    autoComplete="email"
                    autoFocus
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                  />
                  <TextField
                    variant="outlined"
                    margin="normal"
                    required
                    fullWidth
                    name="password"
                    label="Password"
                    type={showPassword ? "text" : "password"}
                    id="password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    InputProps={{
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            aria-label="toggle password visibility"
                            onClick={handleClickShowPassword}
                            onMouseDown={handleMouseDownPassword}
                            edge="end"
                          >
                            {showPassword ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />
                  <Button
                    type="submit"
                    fullWidth
                    variant="contained"
                    color="primary"
                    disabled={isSubmitting}
                    sx={{
                      mt: 3,
                      mb: 2,
                      backgroundColor: "#3B533A",
                    }}
                  >
                    Create account
                  </Button>
                  <Box
                    style={{
                      display: "flex",
                      justifyContent: "center",
                      marginTop: "20px",
                    }}
                  >
                    <Link
                      href="/Login"
                      variant="body2"
                    >
                      Already have an account? Login
                    </Link>
                  </Box>
                </form>
              </Paper>
            </Box>
          </Container>
        </Box>
      </ThemeProvider>
      {isVerificationModalOpen && (
        <VerificationModal
          isOpen={isVerificationModalOpen}
          onClose={closeVerificationModal}
          user={user}
          email={email}
        ></VerificationModal>
      )}
    </div>
  );
}
