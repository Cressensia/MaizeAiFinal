import React, { useState } from "react";
import axios from "axios";
import UploadPic from "../../images/upload-pic.png";
import "./Modal.css";

export default function ModalUploadImage({ isOpen, onClose, onFileUpload }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [images, setImages] = useState([]);

  const handleFileSelect = (e) => {
    setSelectedFiles([...selectedFiles, ...e.target.files]);
  };

  const handleFileRemove = (index) => {
    const newList = selectedFiles.filter((_, idx) => idx !== index);
    setSelectedFiles(newList);
  };

  const handleUpload = async () => {
    console.log("Files to upload:", selectedFiles);
    const filesToUpload = [...selectedFiles];
    await uploadImageToServer(filesToUpload);

    setSelectedFiles([]); // clear selection
    onClose(); // Close the modal after upload
  };

  const handleDrag = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();

    const newFiles = Array.from(e.dataTransfer.files);
    setSelectedFiles([...selectedFiles, ...newFiles]);
  };


  const uploadImageToServer = async () => {
    console.log(images)
    const formData = new FormData();

    await Promise.all(
      selectedFiles.map(async (file, index) => {
        formData.append(`files`, file, file.name);
      })
    );    

    try {
      const response = await axios.post(
        "http://localhost:8000/maizeai/image_upload_view/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const allData = {
        originalImages: selectedFiles.map((file) => ({
          filename: file.name,
          blobUrl: URL.createObjectURL(file),
        })),
        processedResults: response.data.results,
        total: response.data.total_count,
      };

      //setResults(response.data);
      //setProcessedImage(response.data.image_data);
      console.log("Results:", response.data.results);
      console.log("Total count:", response.data.total_count);
      //console.log(`Tassel count: ${response.data.tassel_count}`);

      onFileUpload (allData);
    } catch (e) {
      console.error("Error uploading images:", e);
    }
  };

  const handleCancel = () => {
    setSelectedFiles([]);
    onClose();
  }

  if (!isOpen) return null;

  return (
    <div className="modalOverlay">
      <div className="modal">
        <h2 className="upload-title">Upload images</h2>
        <div
            className="upload-div"
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <label htmlFor="fileInput">
              <div className="image-div">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileSelect}
                  style={{ display: "none" }}
                  id="fileInput"
                />
                <img src={UploadPic} alt="Upload" />
                <p>Drop or upload or images here (max 10 images per upload)</p>
              </div>
            </label>
          </div>
          <div className="filePreviewContainer">
            {selectedFiles.map((file, index) => (
              <div key={index} className="filePreview">
                <img
                  className="image-preview"
                  src={URL.createObjectURL(file)}
                  alt={`Uploaded image ${index + 1}`}
                />
                <div>
                  <span>{file.name}</span>
                  <button onClick={() => handleFileRemove(index)}>Delete</button>
                </div>
              </div>
            ))}
          </div>
        
        <div className="modalActions">
          <button onClick={handleCancel}>Cancel</button>
          <button onClick={handleUpload}>Upload</button>
        </div>
      </div>
    </div>
  );
}