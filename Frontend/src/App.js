import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { AuthProvider } from './AuthContext';
import "./App.css";
import Home from "./components/Home/Home";
import Login from "./components/LoginRegister/Login";
import Dashboard from "./components/Main/Dashboard";
import MaizeCounter from "./components/Main/MaizeCounter";
import Register from "./components/LoginRegister/Register";
import CountWithUs from "./components/Home/CountWithUs";
import WhyMaizeTassels from "./components/Home/WhyMaizeTassels";
import FieldTasselVisualization from "./components/Main/FieldTasselVisualization";
import MaizeDiseaseIdentifier from "./components/Main/MaizeDiseaseIdentifier";
import MaizePhenotypeAnalyzer from "./components/Main/MaizePhenotypeAnalyzer";

export default function App() {
  
  return (
    <div className="App">
      <Router>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/Login" element={<Login />} />
            <Route path="/Dashboard" element={<Dashboard />} />
            <Route path="/MaizeCounter" element={<MaizeCounter />} />
            <Route path="/Register" element={<Register />} />
            <Route path="/CountWithUs" element={<CountWithUs />} />
            <Route path="/WhyMaizeTassels" element={<WhyMaizeTassels />} />
            <Route path="/FieldTasselVisualization" element={<FieldTasselVisualization />} />
            <Route path="/MaizeDiseaseIdentifier" element={<MaizeDiseaseIdentifier />} />
            <Route path="/MaizePhenotypeAnalyzer" element={<MaizePhenotypeAnalyzer />} />
          </Routes>
        </AuthProvider>
      </Router>    
    </div>
  );
}
