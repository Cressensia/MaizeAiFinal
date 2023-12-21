// test

import React, { useState } from "react";
import UserPool from "./UserPool";
import { CognitoUser } from "amazon-cognito-identity-js";

import VerificationModal from "./components/LoginRegister/VerificationModal";

const Signup = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [user, setUser] = useState(null);

//   const onSubmit = (event) => {
//     event.preventDefault();

//     UserPool.signUp(email, password, [], null, (err, data) => {
//       if (err) console.error(err);
//       console.log(data);

//       const cognitoUser = new CognitoUser({
//         Username: data.user.getUsername(),
//         Pool: UserPool,
//       });

//       setUser(cognitoUser);
//       openVerificationModal();
//     });
//   };


  const onSubmit = (event) => {
    event.preventDefault();

    if (!email || !password) {
      alert("Please enter both email and password.");
      return; 
    }

    UserPool.signUp(email, password, [], null, (err, data) => {
      if (err) {
        console.error(err);
        return; 
      }

      const cognitoUser = new CognitoUser({
        Username: data.user.getUsername(), 
        Pool: UserPool,
      });

      setUser(cognitoUser); 
      openVerificationModal();
    });
  };

  const [isVerificationModalOpen, setIsVerificationModalOpen] = useState(false);

  const [isVerificationModalClose, setIsVerificationModalClose] =
    useState(false);

  const openVerificationModal = () => {
    setIsVerificationModalOpen(true);
  };

  const closeVerificationModal = () => {
    setIsVerificationModalOpen(false);
  };

  return (
    <div>
      <form onSubmit={onSubmit}>
        <label htmlFor="email">Email</label>
        <input
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <label htmlFor="password">Password</label>
        <input
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        <button type="submit" >
          Sign Up
        </button>
      </form>
      {isVerificationModalOpen && (
        <VerificationModal
          isOpen={isVerificationModalOpen}
          onClose={closeVerificationModal}
          user={user}
        ></VerificationModal>
      )}
    </div>
  );
};

export default Signup;
