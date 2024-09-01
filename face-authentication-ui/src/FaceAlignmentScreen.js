import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useHistory } from 'react-router-dom';

const FaceAlignmentScreen = () => {
  const [streaming, setStreaming] = useState(false);
  const videoRef = useRef(null);
  const history = useHistory();

  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setStreaming(true);
      });
    }
  }, []);

  const handleNext = () => {
    history.push('/liveness-detection');
  };

  return (
    <div>
      <h1>Align your face within the frame.</h1>
      <video ref={videoRef} width="800" height="600" autoPlay></video>
      <button onClick={handleNext}>Next</button>
    </div>
  );
};

export default FaceAlignmentScreen;
