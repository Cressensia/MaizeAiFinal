// import React, { useState } from "react";

// export default function ModalUploadImage({ isOpen, onClose }) {
//   const [selectedFiles, setSelectedFiles] = useState([]);

//   const handleFileSelect = (event) => {
//     setSelectedFiles([...selectedFiles, ...event.target.files]);
//   };

//   const handleFileRemove = (index) => {
//     const newList = selectedFiles.filter((_, idx) => idx !== index);
//     setSelectedFiles(newList);
//   };

//   const handleUpload = () => {
    
//     console.log("Files to upload:", selectedFiles);
//     // onClose(); // Close the modal after upload
//   };

//   if (!isOpen) return null;

//   return (
//     <div style={styles.modalOverlay}>
//       <div style={styles.modal}>
//         <h2>Upload images</h2>
//         <input
//           type="file"
//           multiple
//           onChange={handleFileSelect}
//           style={styles.fileInput}
//         />
//         <div style={styles.filePreviewContainer}>
//           {selectedFiles.map((file, index) => (
//             <div key={index} style={styles.filePreview}>
//               <span>{file.name}</span>
//               <button onClick={() => handleFileRemove(index)}>Remove</button>
//             </div>
//           ))}
//         </div>
//         <div style={styles.modalActions}>
//           <button onClick={onClose}>Cancel</button>
//           <button onClick={handleUpload}>Upload</button>
//         </div>
//       </div>
//     </div>
//   );
// }

// const styles = {
//   modalOverlay: {
//     position: "fixed",
//     top: 0,
//     left: 0,
//     right: 0,
//     bottom: 0,
//     backgroundColor: "rgba(0, 0, 0, 0.5)",
//     display: "flex",
//     alignItems: "center",
//     justifyContent: "center",
//     zIndex: 1000,
//   },
//   modal: {
//     backgroundColor: "#fff",
//     padding: "20px",
//     borderRadius: "10px",
//     display: "flex",
//     flexDirection: "column",
//     alignItems: "center",
//   },
//   fileInput: {
//     margin: "10px 0",
//   },
//   filePreviewContainer: {
//     alignSelf: "stretch",
//     marginBottom: "10px",
//   },
//   filePreview: {
//     display: "flex",
//     justifyContent: "space-between",
//     alignItems: "center",
//     marginBottom: "5px",
//   },
//   modalActions: {
//     display: "flex",
//     justifyContent: "space-between",
//     width: "100%",
//   },
// };
import React, { useState } from "react";
import axios from "axios";
import UploadPic2 from "../../images/upload-pic2.png";
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
                <img src={UploadPic2} alt="Upload" />
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