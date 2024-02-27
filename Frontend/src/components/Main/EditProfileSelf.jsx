import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Modal,
  ModalDialog,
  DialogTitle,
  DialogContent,
  Button,
  Input,
  FormLabel,
} from "@mui/joy";
import { useAuth } from "../../AuthContext";

function EditProfileSelf({ isOpen, onClose }) {
  const [newName, setNewName] = useState("");
  const { authInfo, setAuthInfo } = useAuth();
  const { authToken, userEmail } = authInfo || {};

  const handleSaveChanges = async () => {
    const formData = new FormData();

    formData.append("email", userEmail);
    formData.append("name", newName);

    try {
      const response = await axios.post(
        `https://api.maizeai.uk/maizeai/manage_user/?email=${userEmail}`,
          formData,
          {
            headers: { 
              "Content-Type": "multipart/form-data",
            },
          }
      );

      console.log("changes saved successfully");
      onClose();  // Close the modal after saving changes
    } catch (e) {
      console.error("error saving changes:", e);
    }
  };

  return (
    <Modal open={isOpen} onClose={onClose}>
      <ModalDialog>
        <DialogTitle>User Profile Edit</DialogTitle>
        <DialogContent>
          <FormLabel>Name</FormLabel>
          <Input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <FormLabel>Email address</FormLabel>
          <Input
          // value={}
          // onChange={(e) => setUsername(e.target.value)}
          />
          <FormLabel>Password</FormLabel>
          <Input
          // required
          // value={fullName}
          // onChange={(e) => setFullName(e.target.value)}
          />
          <Button type="cancel" color="danger" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" color="success"  onClick={handleSaveChanges}>
            Save Changes
          </Button>
        </DialogContent>
      </ModalDialog>
    </Modal>
  );
}

export default EditProfileSelf;