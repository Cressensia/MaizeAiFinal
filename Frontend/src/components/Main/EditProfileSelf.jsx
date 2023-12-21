import React, { useState, useEffect } from "react";
import {
  Modal,
  ModalDialog,
  DialogTitle,
  DialogContent,
  Button,
  Input,
  FormLabel,
} from "@mui/joy";

function EditProfileSelf({ isOpen, onClose }) {
  //   const [accountData, setAccountData] = useState([])
  //   const [username, setUsername] = useState("");
  //   const [fullName, setFullName] = useState("");

  return (
    <Modal open={isOpen} onClose={onClose}>
      <ModalDialog>
        <DialogTitle>User Profile Edit</DialogTitle>
        <DialogContent>
          <FormLabel>Name</FormLabel>
          <Input
          // value={}
          // onChange={(e) => setUsername(e.target.value)}
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
          <Button type="submit" color="success" >Save Changes</Button>
        </DialogContent>
      </ModalDialog>
    </Modal>
  );
}

export default EditProfileSelf;
