import React from 'react';
import { useHistory } from 'react-router-dom';

const CameraPermissionScreen = () => {
  const history = useHistory();

  const grantCameraAccess = () => {
    history.push('/face-alignment');
  };

  return (
    <div>
      <h1>Please allow camera access to start the authentication process.</h1>
      <button onClick={grantCameraAccess}>Grant Camera Access</button>
    </div>
  );
};

export default CameraPermissionScreen;
