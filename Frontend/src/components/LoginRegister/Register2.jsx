//this is a fucked up piece of code cjfiosejkjcselkjcfufifkckckckckkcck grrrrrrrrrrrrrrrrjfhbwkcbskj.jskncfv

import React, { useState } from "react";
import { Link } from "react-router-dom";
import NavbarLogin from "./NavbarLogin";
import "../../App.css";
// import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/joy/Stack";
import FormControl from "@mui/material/FormControl";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import Visibility from "@mui/icons-material/Visibility";
import VisibilityOff from "@mui/icons-material/VisibilityOff";
// import InputLabel from "@mui/material/InputLabel";
// import OutlinedInput from "@mui/material/OutlinedInput";
import TextField from "@mui/material/TextField";
import Button from "@mui/joy/Button";
import VerificationModal from "./VerificationModal";
import CreateAccount from "../../images/CreateAccount.png";
import { CognitoUser } from "amazon-cognito-identity-js";

import "./Register2.css";

import UserPool from "../../UserPool";

export default function Register2() {
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
  // --------------------------

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [user, setUser] = useState(null);

  // const Submit = (event) => {
  //   event.preventDefault();

  //   UserPool.signUp(email, password, [], null, (err, data) => {
  //     if (err) console.error(err);
  //     console.log(data);
  //   });
  // };

  // const onSubmit = (event) => {
  //   event.preventDefault();

  //   UserPool.signUp(email, password, [], null, (err, data) => {
  //     if (err) {
  //       console.error(err);
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

  const onSubmit = (event) => {
    event.preventDefault();

    if (!email || !password) {
      alert("Please enter both email and password.");
      return;
    }

    UserPool.signUp(email, password, [], null, (err, data) => {
      // if (err) {
      //   console.error(err);
      //   return;
      // }
      if (err) {
        if (err.code === "UsernameExistsException") {
          alert("User already exists. Please verify your email.");
          openVerificationModal(); // Reopen the modal for verification
        } else {
          console.error(err);
        }
        return;
      }

      const cognitoUser = new CognitoUser({
        Username: data.user.getUsername(),
        Pool: UserPool,
      });

      setUser(cognitoUser);
      openVerificationModal();
    });
  };

  return (
    <div>
      <NavbarLogin />
      <div className="background">
        <img src={CreateAccount} alt="Create Account" />
        <form onSubmit={onSubmit} className="registration-form">
          <Stack spacing={3} sx={{ m: 1, width: "100%" }}>
            <TextField
              fullWidth
              id="email"
              name="email"
              label="Email"
              variant="filled"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              sx={{
                "& .MuiFilledInput-root": {
                  backgroundColor: "#fff",
                  "&:before": {
                    borderBottom: "none",
                  },
                  "&:hover:before": {
                    borderBottom: "none !important",
                  },
                  "&.Mui-focused": {
                    backgroundColor: "#fff",
                  },
                },
                "& .MuiFilledInput-underline:before": {
                  borderBottom: "none",
                },
                "& .MuiFilledInput-underline:after": {
                  borderBottom: "none",
                },
                borderRadius: "4px",
              }}
            />
            <TextField
              fullWidth
              id="password"
              name="password"
              label="Password"
              type={showPassword ? "text" : "password"}
              variant="filled"
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
              sx={{
                "& .MuiFilledInput-root": {
                  backgroundColor: "#fff",
                  "&:before": {
                    borderBottom: "none",
                  },
                  "&:hover:before": {
                    borderBottom: "none !important",
                  },
                  "&.Mui-focused": {
                    backgroundColor: "#fff", 
                  },
                },
                "& .MuiFilledInput-underline:before": {
                  borderBottom: "none",
                },
                "& .MuiFilledInput-underline:after": {
                  borderBottom: "none", 
                },
                borderRadius: "4px", 
              }}
            />
            <Button
              type="submit"
              fullWidth
              sx={{
                bgcolor: "green",
                ":hover": {
                  bgcolor: "darkgreen",
                },
                borderRadius: "4px",
                color: "white",
              }}
            >
              Create account
            </Button>
          </Stack>
          <Typography
            variant="body1"
            color="white"
            sx={{
              fontFamily: "Inter",
              marginTop: "18px",
              textAlign: "center",
            }}
          >
            <Link
              className="register-link"
              to="/Login"
              style={{ textDecoration: "none", color: "white" }}
            >
              Already have an account?{" "}
              <span style={{ color: "#ACBEFF" }}>Login</span>
            </Link>
          </Typography>
        </form>
      </div>
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
