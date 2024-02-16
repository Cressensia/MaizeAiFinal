import React, { useState, useEffect } from "react";
import axios from "axios";
import NavbarMain from "./NavbarMain";
import Sidebar from "./Sidebar";
import Weather from "../Widget/Weather";

import "./Dashboard.css";

import { useAuth } from "../../AuthContext";
import { Bar } from "react-chartjs-2";
import {
    Chart as ChartJS,
    BarElement,
    CategoryScale,
    LinearScale,
    Tooltip,
    Legend
} from 'chart.js';

import {
    Sheet,
    Table,
    Menu,
    MenuButton,
    MenuItem,
    Dropdown,
    Divider,
  } from "@mui/joy";

ChartJS.register(BarElement,
    CategoryScale,
    LinearScale,
    Tooltip,
    Legend
)

export default function Dashboard() {
    const { authInfo, setAuthInfo } = useAuth();
    const { authToken, userEmail } = authInfo || {};
    const [totalCount, setTotalCount] = useState(0);
    const [monthlyCount, setMonthlyCount] = useState([]);
    const [loading, setLoading] = useState(true);
  

    const fetchTotalCount = async () => {
        try {
        const response = await axios.get(`http://localhost:8000/maizeai/get_total_count/?user_email=${userEmail}`);
        setTotalCount(response.data.total_count);
        } catch (error) {
        console.error('Error fetching total count:', error);
        }
    };

    const fetchMonthlyCount = async () => {
        try {
          const response = await axios.get(
            `http://localhost:8000/maizeai/get_monthly_count/?user_email=${userEmail}`
          );
          setMonthlyCount(response.data.monthly_count);
          setLoading(false);
        } catch (error) {
          console.error("Error fetching monthly count:", error);
          setLoading(false);
        }
      };

    useEffect(() => {
        fetchTotalCount();
        fetchMonthlyCount();
    }, [authToken, userEmail])

    const data = {
        labels: [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ],
        datasets: [
            {             
                backgroundColor: 'rgba(160,210,250,1)',
                borderWidth: 1,
                hoverBackgroundColor: 'rgba(125,158,216,0.4)',
                hoverBorderColor: 'rgba(141,169,218,1)',
                data: monthlyCount,
            },
        ],
    };

    const options = {
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    drawOnChartArea: false
                }   
            },
            x: {
                grid: {   
                    drawOnChartArea: false
                }  
            },
        },
        plugins: {
            legend: {
                display: false,
            },
        },
    };

    return(
        <div className="all">
            <NavbarMain />
            <div className="main2-container">
                <div className="SidebarMain2">
                <Sidebar />
                </div>
                <div className="main2-div">
                <div className="main2-divMaize">
                    <h2>Dashboard</h2>
                    <Divider />               
                    <div className="table">         
                    <h2>Total Maize Count : {totalCount}</h2>         
                        <Table variant="outline" stickyHeader hoverRows>
                            <thead>
                            <tr className="table-header">
                                <div style={{ width: '80%', margin: '0 auto' }}>
                                    <Bar data={data} options={options}/>
                                </div>
                            </tr>
                            </thead>
                        </Table>   
                        <br/><br/><br/>
                        <Divider />
                        <br/><br/><br/>
                        <Weather />  
                        <br/><br/><br/>
                        <Divider />    
                    </div>                           
                </div>
                </div>
            </div>
        </div>
    );
}

/*
7E57C2(purple)
3F9AE9(darker blue )
A0D2FA( light blue)
EFEAF7(light purple)
*/