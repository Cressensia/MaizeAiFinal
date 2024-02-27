import React, { useState } from "react";
import axios from "axios";
import UploadPic2 from "../../images/upload-pic.png";
import "./Modal.css";
import { useAuth } from "../../AuthContext";

export default function ModalUploadImageDisease({ isOpen, onClose, onUploadSuccess }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [images, setImages] = useState([]);
  const { authInfo, setAuthInfo } = useAuth();
  const { authToken, userEmail } = authInfo || {};

  const handleFileSelect = (e) => {
    setSelectedFiles(e.target.files[0] ? [e.target.files[0]] : []);
  };

  const handleFileRemove = (index) => {
    const newList = selectedFiles.filter((_, idx) => idx !== index);
    setSelectedFiles(newList);
  };

  const handleUpload = async () => {
    if (selectedFiles.length > 0) {
      // Upload the single selected file
      await uploadImageToServer(selectedFiles[0]);
      onClose();
      setSelectedFiles([]);
    } else {
      console.error("No file selected for upload.");
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const newFile = e.dataTransfer.files[0];
    setSelectedFiles(newFile ? [newFile] : []);
  };

  const uploadImageToServer = async (file) => {
    const formData = new FormData();
    formData.append("image", file); 

    formData.append("userEmail", userEmail);
    const currentDate = new Date();
    const date = currentDate.toLocaleDateString('en-GB'); // dd/mm/yyyy
    formData.append("uploadDate", date)
  
    try {
      const response = await axios.post(
        "https://api.maizeai.uk/maizeai/upload_imageMaizeDisease/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if(onUploadSuccess) {
        onUploadSuccess();
      }
    } catch (e) {
      console.error("Error uploading image:", e);
    }
  };

  const handleCancel = () => {
    setSelectedFiles([]);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="modalOverlay">
      <div className="modal">
        <h2 className="upload-title">Upload images</h2>
        <div className="upload-div" onDragOver={handleDrag} onDrop={handleDrop}>
          <label htmlFor="fileInput">
            <div className="image-div">
              <input
                type="file"
                accept="image/*"
                // multiple
                onChange={handleFileSelect}
                style={{ display: "none" }}
                id="fileInput"
              />
              <img src={UploadPic2} alt="Upload" class="upload-icon" />
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
