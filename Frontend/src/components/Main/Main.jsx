import React, { useState } from "react";
import NavbarMain from "./NavbarMain";
import UploadPic from "../../images/upload-pic.png";
import "../../App.css";
import "./Main.css";
import axios from "axios";

export default function Main() {
  const [image, setImage] = useState(null);
  const [results, setResults] = useState(null);
  const [formData, setFormData] = useState(null);

  const handleFile = (e) => {
    const file = e.target.files[0];
    const filename = e.target.files[0].name
    console.log(filename)
    if (file) {
      const newFormData = new FormData();
      newFormData.append("file", file);
      
      setImage(URL.createObjectURL(file));
      //console.log("Filename: ", file.name);
      setFormData(newFormData);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0]; // only 1 image
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        // wait for image to load first
        setImage(e.target.result);
        const newFormData = new FormData()
        newFormData.append("file", file);
        setFormData(newFormData);
        uploadImageToServer(newFormData);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleReset = () => {
    setImage(null);
    setResults(null);
    setFormData(null);
  };

  const uploadImageToServer = async () => {
    try {
      //console.log("formData:", formData.entries());

      //now is local, Brenda, u try this on your local, if works ok liao, then we can try to deploy to aws
      const response = await axios.post(
        "http://127.0.0.1:8000/maizeai/upload-image/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      // Assuming the response contains a field 'tassel_count'
      setResults(response.data);
      console.log(results)
      console.log(`Tassel count: ${response.data.tassel_count}`);
    } catch (error) {
      console.error("Error uploading image:", error);
    }
  };

  return (
    <div>
      <NavbarMain />
      <div className="main-container">
        <div className="main-div">
          <div className="main-header">
            <h4>Count Maize Tassels</h4>
          </div>
          <div
            className="upload-div"
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            {image ? (
              <img src={image} alt="Upload Image" />
            ) : (
              <label htmlFor="fileInput">
                <div className="image-div">
                  <input
                    type="file"
                    accept="image/*"
                    multiplay={false}
                    onChange={handleFile}
                    style={{ display: "none" }}
                    id="fileInput"
                  />
                  <img src={UploadPic} alt="Upload" />
                  <p>
                    Click to upload or drag and drop SVN, PNG (MAX, 800×400px)
                  </p>
                </div>
              </label>
            )}
          </div>
          {image && (
            <div className="reset-button">
              <button onClick={handleReset}>Reset Image</button>
              <button onClick={uploadImageToServer}>Process</button>
            </div>
          )}
        </div>
        <div className="main-div">
          <div className="main-header">
            <h4>Results</h4>
          </div>
          <div className="result-div"></div>
          {results && (
            <div>
              <p>Tassel Count: {results.tassel_count}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
