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

import "./VerificationModal.css";

function VerificationModal({ isOpen, onClose, user, email }) {
  const [OTP, setOTP] = useState("");

  // const onSubmit = (event) => {
  //   event.preventDefault();

  //   // console.log(user)
  //   user.confirmRegistration(OTP, true, (err, data) => {
  //     if (err) {
  //       console.log(err);
  //       alert("Couldn't verify account, wrong OTP, try again");
  //     } else {
  //       console.log(data);
  //       alert("Account verified successfully, please log in to continue");
  //       window.location.href = "/Login";
  //     }
  //   });
  // };

  const onSubmit = (event) => {
    event.preventDefault();

    if (user) {
      user.confirmRegistration(OTP, true, (err, data) => {
        if (err) {
          console.error(err);
          alert("Couldn't verify account, wrong OTP, try again");
        } else {
          console.log(data);
          alert("Account verified successfully, please log in to continue");
          window.location.href = "/Login";
        }
      });
    } else {
      console.error("User object is null");
      alert("An unexpected error occurred. Please try again.");
    }
  };

  // const verifyAccount = (e) => {
  //   e.preventDefault();
  //   const user = new CognitoUser({
  //     Username: username,
  //     Pool: UserPool,
  //   });
  //   console.log(user);
  //   user.confirmRegistration(OTP, true, (err, data) => {
  //     if (err) {
  //       console.log(err);
  //       alert("Couldn't verify account");
  //     } else {
  //       console.log(data);
  //       alert('Account verified successfully');
  //       window.location.href = '/login';
  //     }
  //   });
  // };

  return (
    <>
      <Modal open={isOpen} onClose={onClose}>
        <form onSubmit={onSubmit}>
          <ModalDialog className="ModalDialog">
            <DialogTitle className="DialogTitle">Verify your email</DialogTitle>
            <DialogContent className="ModalContent">
              <div>
                <p className="weHave">We have sent the code to your email</p>
                <p className="emailUser">{email}</p>
              </div>
              {/* <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  maxWidth: 300,
                }}
              >
                <Input />
                <Input />
                <Input />
                <Input />
              </div> */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  maxWidth: '100%',
                }}
              >
                <Input
                  className="Input"
                  value={OTP}
                  onChange={(e) => setOTP(e.target.value)}
                  maxLength={6} // Assuming 6-digit OTP
                />
              </div>
              <Button className="Button" type="submit">
                Verify
              </Button>
            </DialogContent>
          </ModalDialog>
        </form>
      </Modal>
    </>
  );
}

export default VerificationModal;
