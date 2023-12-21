import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./components/Home/Home";
import Login from "./components/LoginRegister/Login";
import Main from "./components/Main/Main";
// import Register from "./components/LoginRegister/Register";    use register3
import "./App.css";
import Signup from "./Signup";     //only for testing purpose
// import Register2 from "./components/LoginRegister/Register2"; use register3
import Register3 from "./components/LoginRegister/Register3";

export default function App() {
  return (
    <div className="App">
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Login" element={<Login />} />
          {/* <Route path="/Register" element={<Register />} /> */}
          {/* <Route path="/Register2" element={<Register2 />} /> */}
          <Route path="/Main" element={<Main />} />
          <Route path="/Signup" element={<Signup />} />
          <Route path="/Register3" element={<Register3 />} />
        </Routes>
      </Router>
    </div>
  );
}
