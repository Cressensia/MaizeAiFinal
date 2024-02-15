import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { Dropdown, Menu, MenuButton, MenuItem } from "@mui/joy";
import Avatar from "@mui/joy/Avatar";
import EditProfileSelf from "./EditProfileSelf";
import UserPool from "../../UserPool";
import { useAuth } from "../../AuthContext";
import logo2Main from "../../images/logo2Main.png";
import "./NavbarMain.css";

export default function NavbarMain() {
  const [userInfo, setUserInfo] = useState({});
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isModalClose, setIsModalClose] = useState(false);
  
  const { authInfo, setAuthInfo } = useAuth();
  const { authToken, userEmail } = authInfo || {};
  const navigate = useNavigate();

  const fetchUserDetails = async () => {
    const user = UserPool.getCurrentUser();

    if (user) {
      try {
        const response = await axios.get(
          `http://localhost:8000/maizeai/manage_user/?email=${userEmail}`
        );
        setUserInfo(response.data);
      } catch (e) {
        console.error("Error fetching user details:", e);
      }
    }
  };

  useEffect(() => {
    fetchUserDetails();
  }, [userEmail]);

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

  return (
    <div className="navbarMain">
      <div className="navbarMain-container">
        <Link to="/Dashboard">
          <img className="nav2-logo" src={logo2Main} />
        </Link>
        <Dropdown className="menu">
          <MenuButton>
            <div>
              <Avatar />
            </div>
            {userInfo.name}
          </MenuButton>
          <Menu>
            <MenuItem>
              <div>
                <Avatar />
              </div>
              {userInfo.name}
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
          onClose={() => {
            setIsModalOpen(false);
            fetchUserDetails();
          }}

        >
        </EditProfileSelf>
      )}
    </div>
  );
}
