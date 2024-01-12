import React, { useState } from "react";
import NavbarMain from "./NavbarMain";
import Sidebar from "./Sidebar";
import "./Main2.css";
import { Sheet, Table, Menu, MenuButton, MenuItem, Dropdown, Divider } from "@mui/joy";

import ModalUploadImage from './ModalUploadImage'; 


export default function Main2() {
  const [results, setResults] = useState([]);

  const deleteRecord = (index) => {
    const updatedResults = [...results]; // duplicate current array
    updatedResults.splice(index, 1);
    setResults(updatedResults);
  };

  const updateAssociatedPlots = (result) => {
    // efdsfsdf
  };

  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);

  const handleFileUpload = (allData) => {
    const newResults = allData.originalImages.map((result, index) => ({
      originalImage: result.blobUrl,
      processedImage: allData.processedResults[index].image_data,
      tassel_count: allData.processedResults[index].tassel_count,
      dateOfUpload: new Date().toLocaleDateString(),
      associatedPlots: "", // not yet implemented
      total_count: allData.total,
    }));

    setResults((prevResults) => [...prevResults, ...newResults]);
    closeModal(); // Close the modal after upload
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
            <button className="uploadImageButton " onClick={openModal}>Upload Image</button>
            <Divider />
            <h2>Results</h2>
            <div className="table">
              <Sheet sx={{ height: 400, overflow: "auto", borderRadius: 10, border: '2px solid rgba(0, 0, 0, 0.05)' }}>
                {/* <Sheet > */}
                <Table variant="outline" stickyHeader hoverRows >
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
                    {results.map((result, index) => (
                      <tr key={index}>
                        <td>{index + 1}</td>
                        <td>
                          <img className="result-image"
                            src={result.originalImage}
                            alt={`Original image ${index + 1}`}
                          />
                        </td>
                        <td>{result.dateOfUpload}</td>
                        <td>
                          <img className="result-image"
                            src={`data:image/jpeg;base64, ${result.processedImage}`}
                            alt={`Processed image ${index + 1}`}
                          />
                        </td>
                        <td>{result.tassel_count}</td>
                        <td>{result.associatedPlots}</td>
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
      <ModalUploadImage isOpen={isModalOpen} onClose={closeModal} onFileUpload={handleFileUpload} />
    </div>
  );
}
