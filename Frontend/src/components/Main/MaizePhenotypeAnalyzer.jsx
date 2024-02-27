import React, { useState, useEffect } from 'react';
import axios from "axios";
import NavbarMain from "./NavbarMain";
import Sidebar from "./Sidebar";
import "./Main.css"
import { useAuth } from "../../AuthContext";
import {
    Sheet,
    Table,
    Menu,
    MenuButton,
    MenuItem,
    Dropdown,
    Divider,
} from "@mui/joy";

export default function MaizePhenotypeAnalyzer() {
    const [loading, setLoading] = useState(true);
    const [results, setResults] = useState([]);
    const { authInfo, setAuthInfo } = useAuth();
    const { authToken, userEmail } = authInfo || {};
    const { HSV, setHSV } = useState([])

    const getResultsByEmail = async () => {
        try {
            const response = await axios.get(`https://api.maizeai.uk/maizeai/get_results_by_email/?user_email=${userEmail}`);
            console.log("raw:", response.data);

            const response_data = response.data.counter_results;
            console.log("counter:", response.data.counter_results);
    
            const allOutliers = response_data.reduce((acc, current) => {
                if (current.outliers && Array.isArray(current.outliers)) {
                    return acc.concat(current.outliers);
                }
                return acc;
            }, []);
    
            setResults(allOutliers); 
            console.log(allOutliers);
        } catch (e) {
            console.error("Error fetching results:", e);
        }
    };

    useEffect(() => {
        getResultsByEmail();
    }, [authToken, userEmail]);

    const [previewImage, setPreviewImage] = useState(null);
    const [isPreviewOpen, setIsPreviewOpen] = useState(false);

    const openPreview = (image) => {
        setPreviewImage(image);
        setIsPreviewOpen(true);
    };
    
    const closePreview = () => {
        setPreviewImage(null);
        setIsPreviewOpen(false);
    };

    function HSVtoRGB(hsvArray) {
        let h = hsvArray[0];
        let s = hsvArray[1] / 100;
        let v = hsvArray[2] / 255;
      
        let c = v * s;
        let x = c * (1 - Math.abs(((h / 60) % 2) - 1));
        let m = v - c;
        let r = 0;
        let g = 0;
        let b = 0;
      
        if (h >= 0 && h < 60) {
          r = c; g = x; b = 0;
        } else if (h >= 60 && h < 120) {
          r = x; g = c; b = 0;
        } else if (h >= 120 && h < 180) {
          r = 0; g = c; b = x;
        } else if (h >= 180 && h < 240) {
          r = 0; g = x; b = c;
        } else if (h >= 240 && h < 300) {
          r = x; g = 0; b = c;
        } else if (h >= 300 && h < 360) {
          r = c; g = 0; b = x;
        }
      
        r = Math.round((r + m) * 255);
        g = Math.round((g + m) * 255);
        b = Math.round((b + m) * 255);
      
        return { r, g, b };
      }
      
      // Example usage:
      const hsvArray = [89.51029471134437, 30.411788453774726, 197.3019781994348];
      const rgb = HSVtoRGB(hsvArray);
      console.log(rgb);

    return (
        <div className="all">
        <NavbarMain />
        <div className="main2-container">
            <div className="SidebarMain2">
            <Sidebar />
            </div>
            <div className="main2-div">
            <div className="main2-divMaize">
                <h2>Maize Phenotype Analyzer</h2>
                <Divider />
                <h2>Results</h2>
                <div className="table">
                <Sheet
                    sx={{
                    height: 400,
                    overflow: "auto",
                    borderRadius: 10,
                    border: "2px solid rgba(0, 0, 0, 0.05)",
                    }}
                >
                    <Table variant="outline" stickyHeader hoverRows>
                        <thead>
                        <tr className="table-header">
                            <th>No.</th>
                            <th>Image</th>
                            <th>Dominant Color</th>
                            <th>Color Preview</th>
                            <th>Color Diff</th>
                            <th>Shape Diff</th>
                        </tr>
                        </thead>
                        <tbody className="result-list">
                        {Array.isArray(results) && results.map((result, index) => {
                            const rgb = HSVtoRGB(result.dominant_color); 

                            return (
                                <tr key={index}>
                                <td>{index + 1}</td>
                                <td>
                                    <img                               
                                    src={result.s3_url}
                                    alt={`ROI image ${index + 1}`}
                                    onClick={() => openPreview(result.s3_url)}
                                    style={{ width: '20%', height: 'auto' }}
                                    />
                                </td>
                                <td>
                                    <div>
                                    <div>{`H ${result.dominant_color[0]}`}</div>
                                    <div>{`S ${result.dominant_color[1]}`}</div>
                                    <div>{`V ${result.dominant_color[2]}`}</div>
                                    </div>
                                </td>
                                <td>
                                    <div
                                    style={{
                                        width: '20px',
                                        height: '20px',
                                        backgroundColor: `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`,
                                        borderRadius: '50%'
                                    }}
                                    ></div>
                                </td>
                                <td>
                                    {result.color_diff}
                                </td>
                                <td>
                                    {result.shape_diff}
                                </td>
                                </tr>
                            );
                        })}
                        </tbody>
                    </Table>
                </Sheet>
                </div>
            </div>
            </div>
        </div>
        {isPreviewOpen && (
            <div className="image-preview-modal-overlay" onClick={closePreview}>
            <div
                className="image-preview-modal-content"
                onClick={(e) => e.stopPropagation()}
            >
                <img
                className="image-preview-full"
                src={previewImage}
                alt="Enlarged preview"
                />
                <button className="image-preview-close-btn" onClick={closePreview}>
                ✕
                </button>
            </div>
            </div>
        )}
    </div>
    );
}