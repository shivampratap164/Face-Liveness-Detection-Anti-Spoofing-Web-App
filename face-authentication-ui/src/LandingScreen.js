import React from 'react';
import { useHistory } from 'react-router-dom';

const LandingScreen = () => {
  const history = useHistory();

  const startAuthentication = () => {
    history.push('/camera-permission');
  };

  return (
    <div>
      <h1>Welcome to Aadhaar Face Authentication</h1>
      <p>Ensure secure access by verifying your identity with a simple face scan.</p>
      <button onClick={startAuthentication}>Start Authentication</button>
    </div>
  );
};

export default LandingScreen;
