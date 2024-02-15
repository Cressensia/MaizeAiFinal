import React, { useState, useEffect } from "react";
import axios from "axios";
import NavbarMain from "./NavbarMain";
import Sidebar from "./Sidebar";
import "./FieldTasselVisualization.css";
import { useAuth } from "../../AuthContext";
import GridsForFTV from "./GridsForFTV";
import { Divider } from "@mui/joy";
import { CognitoJwtVerifier } from "aws-jwt-verify";

export default function FieldTasselVisualization() {

  const { authInfo, setAuthInfo } = useAuth();
  const { authToken, userEmail } = authInfo || {};
  const [plots, setPlots] = useState([]);
  const [isTokenValid, setIsTokenValid] = useState(true);

  console.log(authToken);

  useEffect(() => {
    const verifyAndFetchData = async () => {
      const isVerified = await verifyAuthToken();
      console.log("Token verification result:", isVerified);
      if (isVerified) {
        console.log("Token is valid, fetching plots data.");
        fetchPlotsData();
      } else {
        console.log("Token is invalid or has expired.");
      }
    };

    if (userEmail && authToken) {
      verifyAndFetchData();
    } else {
      console.log("userEmail or authToken is missing.");
    }
  }, [userEmail, authToken]);

  const verifyAuthToken = async () => {
    if (!authToken) {
      console.log("No authToken to verify.");
      return false;
    }
    const verifier = CognitoJwtVerifier.create({
      userPoolId: "ap-southeast-1_ORwzbHxDg",
      tokenUse: "id",
      clientId: "3mlrr5116s0sv474g7fapih9rd",
    });

    try {
      const payload = await verifier.verify(authToken);
      console.log("Verified token payload:", payload);
      setIsTokenValid(true);
      return true;
    } catch (error) {
      console.log("Invalid token:", error);
      setIsTokenValid(false);
      return false;
    }
  };

  const fetchPlotsData = async () => {
    if (!userEmail) {
      console.log('No userEmail to fetch data for.');
      return;
    }
    try {
      const response = await axios.get(`http://localhost:8000/maizeai/get_results_by_email/?user_email=${userEmail}`);
      console.log('API response:', response.data.counter_results);
  
      // only those with a plot_name then will be in results
      const filteredResults = response.data.counter_results.filter(result => result.plot_name);
  
      const transformedData = filteredResults.reduce((acc, result) => {
        // Find an existing plot in the accumulator
        let plot = acc.find(p => p.plotName === result.plot_name);
        // If the plot doesn't exist, create it and add to the accumulator
        if (!plot) {
          plot = {
            plotName: result.plot_name,
            sections: []
          };
          acc.push(plot);
        }
        // Push the new section into the plot's sections array
        plot.sections.push({ section: result.section, tasselCount: result.tassel_count });
        return acc;
      }, []);
  
      console.log('Transformed plots data:', transformedData);
      setPlots(transformedData);
    } catch (error) {
      console.error("Fetching plots data failed: ", error);
    }
  };

  return (
    <div className="all">
      <NavbarMain />
      <div className="FieldTasselVisualization-container">
        <div className="SidebarMain2">
          <Sidebar />
        </div>
        <div className="FieldTasselVisualization-div">
          <div className="FieldTasselVisualization-divField">
            <h2>Field Tassel Visualisation</h2>
            <Divider />
            <GridsForFTV plots={plots} />
          </div>
        </div>
      </div>
    </div>
  );
}
