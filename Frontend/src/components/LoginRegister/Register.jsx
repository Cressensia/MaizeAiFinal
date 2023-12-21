// not is use. use Register3 instead

import React, { useState } from "react";
import { Link } from "react-router-dom";
import NavbarLogin from "./NavbarLogin";
import DarkLogo from "../../images/plain-logo-darker.png";
import "../../App.css";
import "./Login.css";
import { createTheme, ThemeProvider, styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/joy/Stack";
import FormControl from "@mui/material/FormControl";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import Visibility from "@mui/icons-material/Visibility";
import VisibilityOff from "@mui/icons-material/VisibilityOff";
import InputLabel from "@mui/material/InputLabel";
import OutlinedInput from "@mui/material/OutlinedInput";
import TextField from "@mui/material/TextField";
import Button from "@mui/joy/Button";

export default function Register() {
  const [showPassword, setShowPassword] = React.useState(false);

  const handleClickShowPassword = () => setShowPassword((show) => !show);

  const handleMouseDownPassword = (event) => {
    event.preventDefault();
  };

  const [registerRequest, setRegisterRequest] = useState({
    email: "",
    password: "",
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    console.log(`Name: ${name}, Value: ${value}`);

    setRegisterRequest((prev) => { 
      return {
        ...prev,
        [name]: value
      };
    });
  };

  const EmailField = styled(TextField)({
    "& label.Mui-focused": {
      color: "#3B533A" // label color
    },
    "& .MuiOutlinedInput-root": {
      "&.Mui-focused fieldset": {
        borderColor: "#3B533A" // outline color
      }
    }
  });

  // change login button default color
  const CreateButton = styled(Button)({
    backgroundColor: "#3B533A",
    "&:hover": {
      backgroundColor: "#5f875e"
    }
  });

  const theme = createTheme({
    palette: {
      primary: {
        main: "#3B533A"
      }
    },
    components: {
      MuiFormControl: {
        styleOverrides: {
          root: {
            backgroundColor: "white" // background color
          }
        }
      }
    }
  });

  const GrayCard = styled(Box)({
    background: "#757975",
    borderRadius: "10px",
    width: "50ch",
    height: "40ch",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    marginTop: "50px",
    padding: "20px"
  });

  const Logo = styled("img")({
    position: "absolute",
    top: "-70px",
    left: "50%",
    transform: "translateX(-50%)",
    borderRadius: "50%",
    backgroundColor: "#FFFFFF",
    padding: "14px"
  });

  return (
    <div>
      <NavbarLogin />
      <div className="login-container">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            const formData = new FormData(event.currentTarget);
            const formJson = Object.fromEntries(formData.entries());
            alert(JSON.stringify(formJson));
          }}
        >
          <GrayCard>
            <Logo src={DarkLogo} alt="logo" />
            <Typography
              variant="h6"
              color="white"
              sx={{
                fontFamily: "Inter",
                textAlign: "center",
                marginBottom: "24px"
              }}
            >
              Create account
            </Typography>
            <Stack spacing={1}>         
              <ThemeProvider theme={theme}>
                {/* <FormControl  // old not working
                  sx={{ m: 1, width: "40ch", borderRadius: "10px" }}
                  variant="outlined"
                >
                  <InputLabel htmlFor="outlined-search">
                    Email
                  </InputLabel>
                  <OutlinedInput
                    id="outlined-search"         
                    name="email"
                    onChange={handleInputChange}
                    value={registerRequest.email}
                    label="email"
                  />
                </FormControl> */}
                  <FormControl
                sx={{ m: 1, width: "40ch", borderRadius: "10px" }}
                variant="outlined"
              >
                <InputLabel htmlFor="outlined-search">
                  Email
                </InputLabel>
                <OutlinedInput
                  id="outlined-search"         
                  name="email"
                  // onChange={handleInputChange}
                  // value={loginRequest.email}
                  label="email"
                />
              </FormControl>
              </ThemeProvider>
              <ThemeProvider theme={theme}>
                {/* <FormControl
                  sx={{ m: 1, width: "40ch", borderRadius: "10px" }}
                  variant="outlined"
                >
                  <InputLabel htmlFor="outlined-adornment-password">
                    Password
                  </InputLabel>
                  <OutlinedInput
                    id="outlined-adornment-password"
                    type={showPassword ? "text" : "password"}
                    name="password"
                    onChange={handleInputChange}
                    value={registerRequest.password}
                    endAdornment={
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
                    }
                    label="Password"
                  />
                </FormControl> */}
                <FormControl
                sx={{ m: 1, width: "40ch", borderRadius: "10px" }}
                variant="outlined"
              >
                <InputLabel htmlFor="outlined-adornment-password">
                  Password
                </InputLabel>
                <OutlinedInput
                  id="outlined-adornment-password"
                  type={showPassword ? "text" : "password"}
                  name="password"
                  // onChange={handleInputChange}
                  // value={loginRequest.password}
                  endAdornment={
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
                  }
                  label="Password"
                />
              </FormControl>
              </ThemeProvider>
              <CreateButton type="submit">
                <Link className="login-link" to="/Main">
                  Create account
                </Link>
              </CreateButton>
            </Stack>
            <Typography
              variant="body1"
              color="white"
              sx={{
                fontFamily: "Inter",
                marginTop: "18px",
                textAlign: "center"
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
          </GrayCard>
        </form>
      </div>
    </div>
  );
}
