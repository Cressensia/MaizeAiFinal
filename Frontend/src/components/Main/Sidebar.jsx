// import * as React from "react";
// import List from "@mui/joy/List";
// import ListItem from "@mui/joy/ListItem";
// import ListItemButton from "@mui/joy/ListItemButton";
// import ListItemDecorator from "@mui/joy/ListItemDecorator";
// import Typography from "@mui/joy/Typography";
// import "./Sidebar.css";
// import DashboardIcon from "@mui/icons-material/Dashboard";
// import AnalyticsIcon from "@mui/icons-material/Analytics";
// import BugReportIcon from "@mui/icons-material/BugReport";
// import VisibilityIcon from "@mui/icons-material/Visibility";
// import Divider from "@mui/joy/Divider";

// export default function Sidebar() {
//   const [open, setOpen] = React.useState(false);

//   return (
//     <React.Fragment>
//       <div className="sidebar">
//         <Typography component="div" className="sidebar-dashboard" style={{ fontWeight: 'bold' }}>
//           Dashboard
//         </Typography>
//         <List>
//           <ListItem>
//             <ListItemButton>
//               <ListItemDecorator>
//                 <DashboardIcon />
//               </ListItemDecorator>
//               <Typography>Dashboard</Typography>
//             </ListItemButton>
//           </ListItem>
//         </List>
//         <Divider />
//         <Typography component="div" className="sidebar-utilities" style={{ fontWeight: 'bold' }} >
//           Utilities
//         </Typography>
//         <List>
//           <ListItem>
//             <ListItemButton>
//               <ListItemDecorator>
//                 <AnalyticsIcon />
//               </ListItemDecorator>
//               <Typography>Maize counter</Typography>
//             </ListItemButton>
//           </ListItem>
//           <ListItem>
//             <ListItemButton>
//               <ListItemDecorator>
//                 <BugReportIcon />
//               </ListItemDecorator>
//               <Typography>Maize Phenotype Analyzer</Typography>
//             </ListItemButton>
//           </ListItem>
//           <ListItem>
//             <ListItemButton>
//               <ListItemDecorator>
//                 <BugReportIcon />
//               </ListItemDecorator>
//               <Typography>Maize Disease Detector</Typography>
//             </ListItemButton>
//           </ListItem>
//           <ListItem>
//             <ListItemButton>
//               <ListItemDecorator>
//                 <VisibilityIcon />
//               </ListItemDecorator>
//               <Typography>Field Tassel Visualization</Typography>
//             </ListItemButton>
//           </ListItem>
//         </List>
//       </div>
//     </React.Fragment>
//   );
// }

import * as React from "react";
import List from "@mui/joy/List";
import ListItem from "@mui/joy/ListItem";
import ListItemButton from "@mui/joy/ListItemButton";
import ListItemDecorator from "@mui/joy/ListItemDecorator";
import ListSubheader from "@mui/material/ListSubheader";
import Typography from "@mui/joy/Typography";
import "./Sidebar.css";
import DashboardIcon from "@mui/icons-material/Dashboard";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import BugReportIcon from "@mui/icons-material/BugReport";
import VisibilityIcon from "@mui/icons-material/Visibility";
import Divider from "@mui/joy/Divider";

export default function Sidebar() {
  const [open, setOpen] = React.useState(false);

  return (
    <React.Fragment>
      <div className="sidebar">
        <Typography
          component="div"
          className="sidebar-dashboard"
          style={{ fontWeight: "bold" }}
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
            <ListItemButton>
              <ListItemDecorator>
                <DashboardIcon />
              </ListItemDecorator>
              <Typography>Dashboard</Typography>
            </ListItemButton>
          </ListItem>
        </List>
        <Divider />
        <Typography
          component="div"
          className="sidebar-utilities"
          style={{ fontWeight: "bold" }}
        >
          Utilities
        </Typography>
        <List>
          <ListItem>
            <ListItemButton>
              <ListItemDecorator>
                <AnalyticsIcon />
              </ListItemDecorator>
              <Typography>Maize counter</Typography>
            </ListItemButton>
          </ListItem>
          <ListItem>
            <ListItemButton>
              <ListItemDecorator>
                <BugReportIcon />
              </ListItemDecorator>
              <Typography>Maize Phenotype Analyzer</Typography>
            </ListItemButton>
          </ListItem>
          <ListItem>
            <ListItemButton>
              <ListItemDecorator>
                <BugReportIcon />
              </ListItemDecorator>
              <Typography>Maize Disease Detector</Typography>
            </ListItemButton>
          </ListItem>
          <ListItem>
            <ListItemButton>
              <ListItemDecorator>
                <VisibilityIcon />
              </ListItemDecorator>
              <Typography>Field Tassel Visualization</Typography>
            </ListItemButton>
          </ListItem>
        </List>
      </div>
    </React.Fragment>
  );
}
