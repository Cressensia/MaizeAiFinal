import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import "./NavbarMain.css";
import logo from "../../images/logo-name-horizontal.png";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import Main from "./Main";
import { Dropdown, Menu, MenuButton, MenuItem } from "@mui/joy";
import Avatar from "@mui/joy/Avatar";
import { useNavigate } from "react-router-dom";
import EditProfileSelf from "./EditProfileSelf";
import { CognitoUser } from "amazon-cognito-identity-js";
import UserPool from "../../UserPool";

export default function NavbarMain() {
  // const [openModal, setOpenModal] = useState(false);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isModalClose, setIsModalClose] = useState(false);

  const openModal = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalClose(false);
  };

  const Logout = () => {
    const user = UserPool.getCurrentUser(); // Get the current user

    if (user) {
      user.signOut(); // Sign out from Cognito
    }
    
    localStorage.clear();
    navigate("/");
  };

  // const [accountData, setAccountData] = useState([])
  const navigate = useNavigate();
  return (
    <div className="navbar">
      <div className="navbar-container">
        <Link to="/Main">
          <img className="nav-logo" src={logo} alt="Logo" />
        </Link>
        <Dropdown className="menu">
          <MenuButton>
            <div>
              <Avatar />
            </div>
            {/* {accountData.full_name} */} Moe Lester
          </MenuButton>
          <Menu>
            <MenuItem>
              <div>
                <Avatar />
              </div>
              {/* {accountData.full_name} */} Moe Lester
              <br></br>
              {/* {profileTypeMap[accountData.p_id]} */} Manager
            </MenuItem>
            <MenuItem onClick={() => setIsModalOpen(true)}>
              Edit Account
            </MenuItem>
            <MenuItem onClick={Logout}>Logout </MenuItem>
          </Menu>
        </Dropdown>
      </div>
      {isModalOpen && (
        <EditProfileSelf
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
        ></EditProfileSelf>
      )}
    </div>
  );
}
