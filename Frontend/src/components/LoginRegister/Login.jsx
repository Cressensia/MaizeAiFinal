// import React, { useState } from "react";
// import { useNavigate, Link } from "react-router-dom"; // check auth success
// import NavbarLogin from "./NavbarLogin";
// import "../../App.css";
// import "./Login.css";
// import { createTheme, ThemeProvider, styled } from "@mui/material/styles";
// import Typography from "@mui/material/Typography";
// import Stack from "@mui/material/Stack";
// import FormControl from "@mui/material/FormControl";
// import InputAdornment from "@mui/material/InputAdornment";
// import IconButton from "@mui/material/IconButton";
// import Visibility from "@mui/icons-material/Visibility";
// import VisibilityOff from "@mui/icons-material/VisibilityOff";
// import InputLabel from "@mui/material/InputLabel";
// import OutlinedInput from "@mui/material/OutlinedInput";
// import TextField from "@mui/material/TextField";
// import Button from "@mui/joy/Button";

// import { CognitoUser, AuthenticationDetails } from "amazon-cognito-identity-js";
// import UserPool from "../../UserPool";

// export default function Login() {
//   const [showPassword, setShowPassword] = React.useState(false);

//   const handleClickShowPassword = () => setShowPassword((show) => !show);

//   const handleMouseDownPassword = (event) => {
//     event.preventDefault();
//   };

//   const navigate = useNavigate();

//   // const [loginRequest, setLoginRequest] = useState({
//   //   email: "",
//   //   password: "",
//   // });

//   // const handleInputChange = (e) => {
//   //   const { name, value } = e.target;
//   //   console.log(`Name: ${name}, Value: ${value}`);

//   //   setLoginRequest((prev) => {
//   //     return {
//   //       ...prev,
//   //       [name]: value
//   //     };
//   //   });
//   // };

//   const [email, setEmail] = useState("");
//   const [password, setPassword] = useState("");

//   const onSubmit = (event) => {
//     event.preventDefault();

//     // if (UserNotConfirmedException) {
//     //   alert("Please verify your email before logging in.");
//     // }

//     const user = new CognitoUser({
//       Username: email,
//       Pool: UserPool,
//     });

//     const authDetails = new AuthenticationDetails({
//       Username: email,
//       Password: password,
//     });

//     user.authenticateUser(authDetails, {
//       onSuccess: (data) => {
//         console.log("onSuccess:", data);
//         navigate("/Main");
//         //need to change to verification modal
//         // nagivate("/VerificationModal")
//       },
//       onFailure: (err) => {
//         console.error("onFailure:", err);
//         alert("Authentication failed: Wrong username or password.");
//       },
//       newPasswordRequired: (data) => {
//         console.log("newPasswordRequired:", data);
//       },
//     });
//   };

//   const EmailField = styled(TextField)({
//     "& label.Mui-focused": {
//       color: "#3B533A", // label color
//     },
//     "& .MuiOutlinedInput-root": {
//       "&.Mui-focused fieldset": {
//         borderColor: "#3B533A", // outline color
//       },
//     },
//   });

//   const LoginButton = styled(Button)({
//     backgroundColor: "#3B533A",
//     "&:hover": {
//       backgroundColor: "#5f875e",
//     },
//   });

//   const theme = createTheme({
//     palette: {
//       primary: {
//         main: "#3B533A",
//       },
//     },
//     components: {
//       MuiFormControl: {
//         styleOverrides: {
//           root: {
//             backgroundColor: "white",
//           },
//         },
//       },
//     },
//   });

//   return (
//     <div>
//       <NavbarLogin />
//       <div className="login-container">
//         <Typography
//           variant="h3"
//           color="white"
//           sx={{
//             fontFamily: "Inter",
//             fontWeight: 100,
//             marginTop: "100px",
//             textAlign: "center",
//           }}
//         >
//           Login to Your Account
//         </Typography>
//         <Typography
//           variant="h6"
//           color="white"
//           sx={{ fontFamily: "Inter", marginTop: "10px", textAlign: "center" }}
//         >
//           The Future of Maize Tassel Monitoring: Your Fields, Your Data,
//           Optimized Crop Yields
//         </Typography>
//         <form
//           // onSubmit={(event) => {
//           //   event.preventDefault();
//           //   const formData = new FormData(event.currentTarget);
//           //   const formJson = Object.fromEntries(formData.entries());
//           //   alert(JSON.stringify(formJson));
//           // }}
//           onSubmit={onSubmit}
//         >
//           <Stack spacing={1}>
//             <ThemeProvider theme={theme}>
//               <FormControl
//                 sx={{ m: 1, width: "40ch"}}
//                 // variant="outlined"
//               >
//                 <InputLabel htmlFor="outlined-search"></InputLabel>
//                 {/* <OutlinedInput */}
//                 <TextField
//                   id="outlined-search"
//                   name="email"
//                   value={email}
//                   onChange={(event) => setEmail(event.target.value)}
//                   label="email"
//                 />
//               </FormControl>
//             </ThemeProvider>
//             <ThemeProvider theme={theme}>
//               <FormControl
//                 sx={{ m: 1, width: "40ch", borderRadius: "10px" }}
//                 variant="outlined"
//               >
//                 <InputLabel htmlFor="outlined-adornment-password">
//                   Password
//                 </InputLabel>
//                 <OutlinedInput
//                   id="outlined-adornment-password"
//                   type={showPassword ? "text" : "password"}
//                   name="password"
//                   value={password}
//                   onChange={(event) => setPassword(event.target.value)}
//                   endAdornment={
//                     <InputAdornment position="end">
//                       <IconButton
//                         aria-label="toggle password visibility"
//                         onClick={handleClickShowPassword}
//                         onMouseDown={handleMouseDownPassword}
//                         edge="end"
//                       >
//                         {showPassword ? <VisibilityOff /> : <Visibility />}
//                       </IconButton>
//                     </InputAdornment>
//                   }
//                   label="Password"
//                 />
//               </FormControl>
//             </ThemeProvider>
//             <LoginButton type="submit">
//               <Link className="login-link">
//                 {" "}
//                 {/*  to="/Main"     to verification modal  */}
//                 Login
//               </Link>
//             </LoginButton>
//           </Stack>
//           <Typography
//             variant="body1"
//             color="white"
//             sx={{ fontFamily: "Inter", marginTop: "18px", textAlign: "center" }}
//           >
//             <Link
//               className="register-link"
//               to="/Register2"
//               style={{ textDecoration: "none", color: "white" }}
//             >
//               New user? Create an account here!
//             </Link>
//           </Typography>
//           <Typography
//             variant="body1"
//             color="white"
//             sx={{ fontFamily: "Inter", textAlign: "center" }}
//           >
//             Forgot Password?
//           </Typography>
//         </form>
//       </div>
//     </div>
//   );
// }

import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom"; // check auth success
import NavbarLogin from "./NavbarLogin";
import "../../App.css";
import "./Login.css";
import { createTheme, ThemeProvider, styled } from "@mui/material/styles";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import FormControl from "@mui/material/FormControl";
import InputAdornment from "@mui/material/InputAdornment";
import IconButton from "@mui/material/IconButton";
import Visibility from "@mui/icons-material/Visibility";
import VisibilityOff from "@mui/icons-material/VisibilityOff";
import InputLabel from "@mui/material/InputLabel";
import OutlinedInput from "@mui/material/OutlinedInput";
import TextField from "@mui/material/TextField";
import Button from "@mui/joy/Button";

import { CognitoUser, AuthenticationDetails } from "amazon-cognito-identity-js";
import UserPool from "../../UserPool";

export default function Login() {
  const [showPassword, setShowPassword] = React.useState(false);

  const handleClickShowPassword = () => setShowPassword((show) => !show);

  const handleMouseDownPassword = (event) => {
    event.preventDefault();
  };

  const navigate = useNavigate();

  // const [loginRequest, setLoginRequest] = useState({
  //   email: "",
  //   password: "",
  // });

  // const handleInputChange = (e) => {
  //   const { name, value } = e.target;
  //   console.log(`Name: ${name}, Value: ${value}`);

  //   setLoginRequest((prev) => {
  //     return {
  //       ...prev,
  //       [name]: value
  //     };
  //   });
  // };

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const onSubmit = (event) => {
    event.preventDefault();

    // if (UserNotConfirmedException) {
    //   alert("Please verify your email before logging in.");
    // }

    const user = new CognitoUser({
      Username: email,
      Pool: UserPool,
    });

    const authDetails = new AuthenticationDetails({
      Username: email,
      Password: password,
    });

    user.authenticateUser(authDetails, {
      onSuccess: (data) => {
        console.log("onSuccess:", data);
        navigate("/Main");
        //need to change to verification modal
        // nagivate("/VerificationModal")
      },
      onFailure: (err) => {
        console.error("onFailure:", err);
        alert("Authentication failed: Wrong username or password.");
      },
      newPasswordRequired: (data) => {
        console.log("newPasswordRequired:", data);
      },
    });
  };

  const EmailField = styled(TextField)({
    "& label.Mui-focused": {
      color: "#3B533A", // label color
    },
    "& .MuiOutlinedInput-root": {
      "&.Mui-focused fieldset": {
        borderColor: "#3B533A", // outline color
      },
    },
  });

  const LoginButton = styled(Button)({
    backgroundColor: "#3B533A",
    "&:hover": {
      backgroundColor: "#5f875e",
    },
  });

  const theme = createTheme({
    palette: {
      primary: {
        main: "#3B533A",
      },
    },
    components: {
      MuiFormControl: {
        styleOverrides: {
          root: {
            backgroundColor: "white",
          },
        },
      },
    },
  });

  return (
    <div>
      <NavbarLogin />
      <div className="login-container">
        <Typography
          variant="h3"
          color="white"
          sx={{
            fontSize: "40px",
            fontFamily: "Inter",
            fontWeight: 900,
            marginTop: "100px",
            textAlign: "center",
          }}
        >
          Login to Your Account
        </Typography>
        <Typography
          variant="h6"
          color="white"
          sx={{ fontFamily: "Inter", marginTop: "10px", textAlign: "center", fontSize: "19px"}}
        >
          The Future of Maize Tassel Monitoring: Your Fields, Your Data,
          Optimized Crop Yields
        </Typography>
        <form
          // onSubmit={(event) => {
          //   event.preventDefault();
          //   const formData = new FormData(event.currentTarget);
          //   const formJson = Object.fromEntries(formData.entries());
          //   alert(JSON.stringify(formJson));
          // }}
          onSubmit={onSubmit}
        >
          <Stack spacing={1}>
            <ThemeProvider theme={theme}>
              <FormControl variant="filled" sx={{ m: 1, width: "40ch" }}>
                <TextField
                  id="outlined-search"
                  name="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  label="Email"
                  variant="filled"
                  sx={{
                    "& .MuiFilledInput-underline:after": {
                      borderBottom: "none",
                    },
                    "& .MuiFilledInput-underline:before": {
                      borderBottom: "none",
                    },
                    "&:hover .MuiFilledInput-underline:before": {
                      borderBottom: "none",
                    },
                    "& .MuiFilledInput-root": {
                      backgroundColor: "transparent",
                      borderRadius: "4px",
                      "&:hover": {
                        backgroundColor: "transparent",
                      },
                      "&.Mui-focused": {
                        backgroundColor: "transparent",
                        boxShadow: "0 0 0 2px #3B533A",
                      },
                    },
                  }}
                />
              </FormControl>
            </ThemeProvider>
            <ThemeProvider theme={theme}>
              <FormControl variant="filled" sx={{ m: 1, width: "40ch" }}>
                <InputLabel htmlFor="outlined-adornment-password">
                  Password
                </InputLabel>
                <OutlinedInput
                  id="outlined-adornment-password"
                  type={showPassword ? "text" : "password"}
                  name="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
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
                  sx={{
                    "& .MuiOutlinedInput-notchedOutline": {
                      border: "none",
                    },
                    "&:hover .MuiOutlinedInput-notchedOutline": {
                      border: "none",
                    },
                    "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                      border: "none",
                    },
                    "&.MuiOutlinedInput-root": {
                      borderRadius: "4px", // Adjust this value to increase or decrease the roundness
                      "&:hover": {
                        borderColor: "none",
                      },
                      "&.Mui-focused": {
                        boxShadow: "0 0 0 2px #3B533A", // Optional: if you want to add focus ring
                      },
                    },
                  }}
                />
              </FormControl>
            </ThemeProvider>
            {/* <LoginButton type="submit">
              <Link className="login-link">Login</Link>
            </LoginButton> */}
            <LoginButton type="submit">Login</LoginButton>
          </Stack>
          <Typography
            variant="body1"
            color="white"
            sx={{ fontFamily: "Inter", marginTop: "18px", textAlign: "center" }}
          >
            <Link
              className="register-link"
              to="/Register3"
              style={{ textDecoration: "none", color: "white" }}
            >
              New user? Create an account here!
            </Link>
          </Typography>
          <Typography
            variant="body1"
            color="white"
            sx={{ fontFamily: "Inter", textAlign: "center" }}
          >
            Forgot Password?
          </Typography>
        </form>
      </div>
    </div>
  );
}
