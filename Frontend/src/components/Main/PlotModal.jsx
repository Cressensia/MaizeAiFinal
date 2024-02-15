import React, { useState } from "react";
import axios from "axios";
import "./Modal.css";

export default function PlotModal({ isOpen, onClose, onUpdatePlots, result }) {
  const [plotName, setPlotName] = useState("");
  const [selectedSection, setSelectedSection] = useState("");

  const handleUpdate = async () => {
    const formData =  new FormData();

    formData.append("documentId", result.document_id);
    formData.append("plotName", plotName);
    formData.append("section", selectedSection);

    try {
      const response = await axios.post(
        "http://localhost:8000/maizeai/update_plots/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      onUpdatePlots(response.data.updatedResults);

      onClose();
    } catch (e) {
      console.error("Error updating associated plots:", e);
    }
  };

  const handleCancel = () => {
    setPlotName("");
    setSelectedSection("");
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="modalOverlay">
      <div className="modal">
        <h2 className="update-title">Update Associated Plots</h2>
        <div className="form-group">
          <label htmlFor="plotName">Associated Plot Name:</label>
          <input
            type="text"
            id="plotName"
            value={plotName}
            onChange={(e) => setPlotName(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="section">Section:</label>
          <select
            id="section"
            value={selectedSection}
            onChange={(e) => setSelectedSection(e.target.value)}
          >
            <option value="">Select Section</option>
            <option value="1">Section 1</option>
            <option value="2">Section 2</option>
            <option value="3">Section 3</option>
            <option value="4">Section 4</option>
            <option value="5">Section 5</option>
            <option value="6">Section 6</option>
          </select>
        </div>
        <div className="modalActions">
          <button onClick={handleCancel}>Cancel</button>
          <button onClick={handleUpdate}>Update</button>
        </div>
      </div>
    </div>
  );
}
