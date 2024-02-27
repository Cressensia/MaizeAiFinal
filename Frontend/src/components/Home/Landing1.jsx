import React from "react";
import "./Home.css";
import MaizeBackground from "../../images/maize-landing.jpg";
import PlainLogo from "../../images/plain-logo.png";

export default function Landing1() {
  return (
    <div className="landingone-container">
      <img src={MaizeBackground} />
      <div className="landingone-div">
        <img className="landingone-logo" src={PlainLogo} />
        <h1 className="landingone-heading">We detect and count</h1>
        <h1 className="landingone-heading">Maize Tassels</h1>
      </div>
    </div>
  );
}
