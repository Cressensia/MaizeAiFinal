import React, { useState, useEffect } from "react";
import axios from "axios";
import NavbarMain from "./NavbarMain";
import Sidebar from "./Sidebar";
import "./MaizeDiseaseIdentifier.css";
import {
  Sheet,
  Table,
  Menu,
  MenuButton,
  MenuItem,
  Dropdown,
  Divider,
} from "@mui/joy";

import { useAuth } from "../../AuthContext";
import ModalUploadImageDisease from "./ModalUploadImageDisease";

export default function MaizeDiseaseIdentifier() {
  const [results, setResults] = useState([]);
  const { authInfo, setAuthInfo } = useAuth();
  console.log("Initial authInfo:", authInfo);
  const { authToken, userEmail } = authInfo || {};

  const deleteRecord = async (index) => {
    try {
      const documentId = results[index].document_id;
      await axios.delete(`http://localhost:8000/maizeai/delete_record/disease/${documentId}`);
      
      const updatedResults = results.filter((_, i) => i !== index);
      setResults(updatedResults);
    } catch (e) {
      console.error("Error deleting record:", e);
    }
  };

  const getResultsByEmail = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/maizeai/get_results_by_email/?user_email=${userEmail}`);
      setResults(response.data.disease_results);
    } catch (e) {
      console.error("Error fetching results:", e);
    }
  };

  useEffect(() => {
    getResultsByEmail();
  }, [authToken, userEmail]);

  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);

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

  const openUpdateModal = (index) => {
    setSelectedResultIndex(index);
    setIsModalOpen(true);
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
            <h2>Maize Disease Detector</h2>
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
                <Table variant="outline" stickyHeader hoverRows>
                  <thead>
                    <tr className="table-header">
                      <th>No.</th>
                      <th>Image</th>
                      <th>Date of upload</th>
                      <th>Maize Disease Results</th>
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
                            onClick={() => openPreview(result.image)}
                          />
                        </td>
                        <td>{result.upload_date}</td>
                        <td>{result.disease_type}</td>
                        <td>
                          <Dropdown>
                            <MenuButton>...</MenuButton>
                            <Menu>
                              <MenuItem onClick={() => deleteRecord(index)}>
                                Delete Record
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
      <ModalUploadImageDisease
        isOpen={isModalOpen}
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
    </div>
  );
}
