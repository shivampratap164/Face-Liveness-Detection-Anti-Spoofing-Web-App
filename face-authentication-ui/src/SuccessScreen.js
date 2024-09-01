import React from 'react';
import { useHistory } from 'react-router-dom';

const SuccessScreen = () => {
  const history = useHistory();

  const continueToNextStep = () => {
    // Redirect to another page or log out
  };

  return (
    <div>
      <h1>Authentication Successful!</h1>
      <button onClick={continueToNextStep}>Continue</button>
    </div>
  );
};

export default SuccessScreen;
