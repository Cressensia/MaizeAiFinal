import React, { useState } from "react";
import { Link } from "react-router-dom";
import "./NavbarLanding.css";
import logo from "../../images/logo-name-horizontal.png";
import MenuIcon from "@mui/icons-material/Menu";

export default function Navbar() {
  const [click, setClick] = useState(false);

  const handleClick = () => setClick(!click);
  const closeMobileMenu = () => setClick(false);

  return (
    <div>
      <div className="navbar">
        <div className="navbar-container">
          <Link to="/" onClick={closeMobileMenu}>
            <img className="nav-logo" src={logo} />
          </Link>
          <div className="menu-icon" onClick={handleClick}>
            <MenuIcon />
          </div>
          <ul className={click ? "nav-menu active" : "nav-menu"}>
            <li className="nav-item">
              <Link to="/WhyMaizeTassels" className="nav-links" onClick={closeMobileMenu}>
                Why Maize Tassels
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/CountWithUs" className="nav-links" onClick={closeMobileMenu}>
                Count With Us
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/" className="nav-links" onClick={closeMobileMenu}>
                Contact Us
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/Login" className="nav-links" onClick={closeMobileMenu}>
                Login
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
