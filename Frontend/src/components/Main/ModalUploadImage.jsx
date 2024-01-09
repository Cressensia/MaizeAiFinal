import React, { useState } from "react";

export default function ModalUploadImage({ isOpen, onClose }) {
  const [selectedFiles, setSelectedFiles] = useState([]);

  const handleFileSelect = (event) => {
    setSelectedFiles([...selectedFiles, ...event.target.files]);
  };

  const handleFileRemove = (index) => {
    const newList = selectedFiles.filter((_, idx) => idx !== index);
    setSelectedFiles(newList);
  };

  const handleUpload = () => {
    
    console.log("Files to upload:", selectedFiles);
    // onClose(); // Close the modal after upload
  };

  if (!isOpen) return null;

  return (
    <div style={styles.modalOverlay}>
      <div style={styles.modal}>
        <h2>Upload images</h2>
        <input
          type="file"
          multiple
          onChange={handleFileSelect}
          style={styles.fileInput}
        />
        <div style={styles.filePreviewContainer}>
          {selectedFiles.map((file, index) => (
            <div key={index} style={styles.filePreview}>
              <span>{file.name}</span>
              <button onClick={() => handleFileRemove(index)}>Remove</button>
            </div>
          ))}
        </div>
        <div style={styles.modalActions}>
          <button onClick={onClose}>Cancel</button>
          <button onClick={handleUpload}>Upload</button>
        </div>
      </div>
    </div>
  );
}

const styles = {
  modalOverlay: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
  },
  modal: {
    backgroundColor: "#fff",
    padding: "20px",
    borderRadius: "10px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  fileInput: {
    margin: "10px 0",
  },
  filePreviewContainer: {
    alignSelf: "stretch",
    marginBottom: "10px",
  },
  filePreview: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "5px",
  },
  modalActions: {
    display: "flex",
    justifyContent: "space-between",
    width: "100%",
  },
};
