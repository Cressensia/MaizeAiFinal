import React, { useState, useEffect } from "react";
import { CognitoUser } from "amazon-cognito-identity-js";
import {
  Modal,
  ModalDialog,
  DialogTitle,
  DialogContent,
  Button,
  Input,
  FormLabel,
  Textarea,
} from "@mui/joy";

import "./ResetPasswordModal.css";

function ResetPasswordModal({ isOpen, onClose, user, email }) {
  return (
    <>
      <Modal open={isOpen} onClose={onClose}>
        <form onSubmit={onSubmit}>
          <ModalDialog className="ModalDialog">
            <DialogTitle className="DialogTitle">
              Reset your password
            </DialogTitle>
            <DialogContent className="ModalContent">
              <div>
                <p>Enter your email to reset your password</p>
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  maxWidth: 300,
                }}
              >
                <Input
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
              <Button type="submit">Send reset link</Button>
            </DialogContent>
          </ModalDialog>
        </form>
      </Modal>
    </>
  );
}

export default ResetPasswordModal;
