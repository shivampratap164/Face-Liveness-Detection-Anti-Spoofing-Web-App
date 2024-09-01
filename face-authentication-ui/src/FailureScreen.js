import React from 'react';
import { useHistory } from 'react-router-dom';

const FailureScreen = () => {
  const history = useHistory();

  const retry = () => {
    history.push('/face-alignment');
  };

  return (
    <div>
      <h1>Authentication Failed</h1>
      <button onClick={retry}>Retry</button>
    </div>
  );
};

export default FailureScreen;
