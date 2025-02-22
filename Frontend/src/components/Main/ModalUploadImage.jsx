import React, { useState } from "react";
import axios from "axios";
import UploadPic from "../../images/upload-pic.png";
import "./Modal.css";
import { useAuth } from "../../AuthContext";

export default function ModalUploadImage({ isOpen, onClose, onUploadSuccess }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [images, setImages] = useState([]);
  const [isUploading, setIsUploading] = useState(false); // New state variable to track uploading status

  const { authInfo, setAuthInfo } = useAuth();
  const { authToken, userEmail } = authInfo || {};

  const handleFileSelect = (e) => {
    setSelectedFiles([...selectedFiles, ...e.target.files]);
  };

  const handleFileRemove = (index) => {
    const newList = selectedFiles.filter((_, idx) => idx !== index);
    setSelectedFiles(newList);
  };

  const handleUpload = async () => {
    setIsUploading(true); // Disable the upload button
    console.log("Files to upload:", selectedFiles);
    const filesToUpload = [...selectedFiles];
    await uploadImageToServer(filesToUpload);

    setSelectedFiles([]); // clear selection
    onClose(); // Close the modal after upload
    setIsUploading(false); // Re-enable the upload button
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
    console.log(images);
    const formData = new FormData();

    await Promise.all(
      selectedFiles.map(async (file, index) => {
        formData.append(`files`, file, file.name);
      })
    );

    formData.append("userEmail", userEmail);
    const currentDate = new Date();
    const date = currentDate.toLocaleDateString("en-GB"); // dd/mm/yyyy
    formData.append("uploadDate", date);

    try {
      const response = await axios.post(
        "https://count.maizeai.uk/modelapp/image_upload_view/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: authToken,
          },
        }
      );

      if (onUploadSuccess) {
        onUploadSuccess();
      }
    } catch (e) {
      console.error("Error uploading images:", e);
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
                  multiple
                  onChange={handleFileSelect}
                  style={{ display: "none" }}
                  id="fileInput"
                />
                <img src={UploadPic} alt="Upload" class="upload-icon" />
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
            <button onClick={handleUpload} disabled={isUploading}>
              Upload
            </button>{" "}
            {/* Disable button based on isUploading */}
          </div>
        </div>
      </div>
    );
  }

