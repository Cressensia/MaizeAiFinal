import React, { useState, useEffect } from "react";
import axios from "axios";
import { useLocation } from "react-router-dom";
import NavbarMain from "./NavbarMain";
import Sidebar from "./Sidebar";
import "./MaizeCounter.css";
import "./Main.css"
import {
  Sheet,
  Table,
  Menu,
  MenuButton,
  MenuItem,
  Dropdown,
  Divider,
} from "@mui/joy";

import ModalUploadImage from "./ModalUploadImage";
import PlotModal from "./PlotModal";
import { CognitoJwtVerifier } from "aws-jwt-verify";
import { useAuth } from "../../AuthContext";

export default function MaizeCounter() {
  const [results, setResults] = useState([]);
  const location = useLocation();
  const [isTokenValid, setIsTokenValid] = useState(true);
  const { authInfo, setAuthInfo } = useAuth();
  const { authToken, userEmail } = authInfo || {};
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [selectedResult, setSelectedResult] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);

  const verifyAuthToken = async () => {
    const verifier = CognitoJwtVerifier.create({
      userPoolId: "ap-southeast-1_ORwzbHxDg",
      tokenUse: "id",
      clientId: "3mlrr5116s0sv474g7fapih9rd",
    });

    try {
      const payload = await verifier.verify(authToken);
      setIsTokenValid(true);
    } catch (error) {
      console.log("Invalid token:", error);
      setIsTokenValid(false);
    }
  };

  const getResultsByEmail = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/maizeai/get_results_by_email/?user_email=${userEmail}`);

      setResults(response.data.counter_results);
    } catch (e) {
      console.error("Error fetching results:", e);
    }
  };

  useEffect(() => {
    verifyAuthToken();
    getResultsByEmail();
    console.log(authToken, userEmail);
  }, [authToken, userEmail]);

  const deleteRecord = async (index) => {
    try {
      const documentId = results[index].document_id;
      await axios.delete(`http://localhost:8000/maizeai/delete_record/counter/${documentId}`);
      
      const updatedResults = results.filter((_, i) => i !== index);
      setResults(updatedResults);
    } catch (e) {
      console.error("Error deleting record:", e);
    }
  };

  const updateAssociatedPlots = (result) => {
    setSelectedResult(result);
    setIsUpdateModalOpen(true);
  };

  //preview image
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
            <h2>Maize Counter</h2>
            <button className="uploadImageButton " onClick={openModal}>
              Upload Image
            </button>
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
                {/* <Sheet > */}
                <Table variant="outline" stickyHeader hoverRows>
                  <thead>
                    <tr className="table-header">
                      <th>No.</th>
                      <th>Image</th>
                      <th>Date of upload</th>
                      <th>Results</th>
                      <th>Tassel Counts</th>
                      <th>Associated plots</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody className="result-list">
                    {Array.isArray(results) && results.map((result, index) => (
                      <tr key={index}>
                        <td>{index + 1}</td>
                        <td>
                          <img
                            className="result-image"
                            src={result.original_image}
                            alt={`Original image ${index + 1}`}
                            onClick={() => openPreview(result.original_image)}
                          />
                        </td>
                        <td>{result.upload_date}</td>
                        <td>
                          <img
                            className="result-image"
                            src={result.processed_image}
                            alt={`Processed image ${index + 1}`}
                            onClick={() => openPreview(result.processed_image)}
                          />
                        </td>
                        <td>{result.tassel_count}</td>
                        <td>{result.plot_name || result.section ? `${result.plot_name}, ${result.section}` : 'nil'}</td>
                        <td>
                          <Dropdown>
                            <MenuButton>...</MenuButton>
                            <Menu>
                              <MenuItem onClick={() => deleteRecord(index)}>
                                Delete Record
                              </MenuItem>
                              <MenuItem
                                onClick={() => updateAssociatedPlots(result)}
                              >
                                Update associated plots
                              </MenuItem>
                            </Menu>
                          </Dropdown>
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
      <ModalUploadImage
        isOpen={isModalOpen && isTokenValid}
        onClose={closeModal}
        onUploadSuccess={getResultsByEmail}
      />
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
      <PlotModal
        isOpen={isUpdateModalOpen}
        onClose={() => setIsUpdateModalOpen(false)}
        onUpdatePlots={(updatedResults) => {
          setResults(updatedResults);
          getResultsByEmail();
        }}
        result={selectedResult}
      />
    </div>
  );
}
