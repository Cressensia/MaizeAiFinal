import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [authInfo, setAuthInfo] = useState({
    authToken: null,
    userEmail: null,
  });

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedToken = localStorage.getItem('authToken');
    const storedUserEmail = localStorage.getItem('userEmail');

    if (storedToken && storedUserEmail) {
      setAuthInfo({
        authToken: storedToken,
        userEmail: storedUserEmail,
      });
    }

    setLoading(false);
  }, []);

  return (
    <AuthContext.Provider value={{ authInfo, setAuthInfo, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};