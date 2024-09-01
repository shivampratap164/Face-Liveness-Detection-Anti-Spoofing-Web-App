import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useHistory } from 'react-router-dom';

const LivenessDetectionScreen = () => {
  const [streaming, setStreaming] = useState(false);
  const [result, setResult] = useState(null);
  const videoRef = useRef(null);
  const history = useHistory();

  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setStreaming(true);
        startLivenessDetection();
      });
    }
  }, []);

  const startLivenessDetection = () => {
    // Assuming you have an endpoint to handle liveness detection
    axios.get('/api/start-video').then((response) => {
      setResult(response.data.result); // Set result from API
    });
  };

  const handleFinish = () => {
    if (result === 'success') {
      history.push('/success');
    } else {
      history.push('/failure');
    }
  };

  return (
    <div>
      <h1>Please blink your eyes slowly.</h1>
      <video ref={videoRef} width="800" height="600" autoPlay></video>
      <button onClick={handleFinish}>Finish</button>
    </div>
  );
};

export default LivenessDetectionScreen;
