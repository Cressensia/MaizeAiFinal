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

    const getResultsByEmail = async () => {
        try {
            const response = await axios.get(`http://localhost:8000/maizeai/get_results_by_email/?user_email=${userEmail}`);
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
                        {Array.isArray(results) && results.map((result, index) => (
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
                                    <div>{`R ${result.dominant_color[0]}`}</div>
                                    <div>{`G ${result.dominant_color[1]}`}</div>
                                    <div>{`B ${result.dominant_color[2]}`}</div>
                                </div>
                            </td>
                            <td>
                                <div
                                style={{
                                    width: '20px',
                                    height: '20px',
                                    backgroundColor: `rgb(${result.dominant_color.join(", ")})`,
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
                        ))}
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