import React, { useState } from "react";
import { Link } from 'react-router-dom';

import "./Sidebar.css";
import dashboardIcon from "../../images/dashboard.png";
import A from "../../images/A.png";
import paint from "../../images/paint.png";
import moon from "../../images/moon.png";
import fan from "../../images/fan.png";

import List from "@mui/joy/List";
import ListItem from "@mui/joy/ListItem";
import ListItemButton from "@mui/joy/ListItemButton";
import ListItemDecorator from "@mui/joy/ListItemDecorator";
import ListSubheader from "@mui/material/ListSubheader";
import Typography from "@mui/joy/Typography";
import Divider from "@mui/joy/Divider";

export default function Sidebar() {

  const [isOpen, setIsOpen] = useState(true);

  const toggle = () => {
    setIsOpen(!isOpen);
    // console.log("Sidebar state: ", isOpen);
  }; 

  const sidebarStyle = {
    width: isOpen ? "210px" : "50px", 
    transition: "width 0.5s",
  };

  return (
    <React.Fragment>
      <div className="sidebar" style={sidebarStyle}>
      <button onClick={toggle} className="sidebar-toggle">{isOpen ? 'Close' : 'Open'} </button>
        <Typography
          component="div"
          className="sidebar-dashboard"
          // style={{ fontWeight: "bold" }}
        >
          Dashboard
        </Typography>
        <List
          sx={{ width: "100%", maxWidth: 360, bgcolor: "background.paper" }}
          component="nav"
          aria-labelledby="nested-list-subheader"
          subheader={
            <ListSubheader component="div" id="nested-list-subheader">
              Nested List Items
            </ListSubheader>
          }
        >
          <ListItem>
            <Link to ="/Dashboard" className="link">
              <ListItemButton>             
                  <ListItemDecorator>
                    <img src={dashboardIcon} alt="dashboard" />
                  </ListItemDecorator>            
                <Typography>Dashboard</Typography>
              </ListItemButton>
            </Link>
          </ListItem>
        </List>
        <Divider />
        <Typography
          component="div"
          className="sidebar-utilities"
        >
          Utilities
        </Typography>
        <List>
          <ListItem>
            <Link to ="/MaizeCounter" className="link">
              <ListItemButton>
                <ListItemDecorator>
                  {/* <AnalyticsIcon /> */}
                  <img src={A} alt="A" />
                </ListItemDecorator>
                <Typography>Maize counter</Typography>
              </ListItemButton>
            </Link>
          </ListItem>
          <ListItem>
            <Link to ="/MaizePhenotypeAnalyzer" className="link">
              <ListItemButton>
                <ListItemDecorator>
                  {/* <BugReportIcon /> */}
                  <img src={paint} alt="paint" />
                </ListItemDecorator>
                <Typography>Maize Phenotype Analyzer</Typography>
              </ListItemButton>
            </Link>
          </ListItem>
          <ListItem>
            <Link to ="/MaizeDiseaseIdentifier" className="link">
              <ListItemButton>
                <ListItemDecorator>
                  {/* <BugReportIcon /> */}
                  <img src={moon} alt="moon" />
                </ListItemDecorator>
                <Typography>Maize Disease Detector</Typography>
              </ListItemButton>
            </Link>
          </ListItem>
          <ListItem>
          <Link to ="/FieldTasselVisualization" className="link">
            <ListItemButton>
              <ListItemDecorator>
                {/* <VisibilityIcon /> */}
                <img src={fan} alt="fan" />
              </ListItemDecorator>
              <Typography>Field Tassel Visualization</Typography>
            </ListItemButton>
            </Link>
          </ListItem>
        </List>
      </div>
    </React.Fragment>
  );
}


