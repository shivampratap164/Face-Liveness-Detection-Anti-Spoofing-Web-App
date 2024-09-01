import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import LandingScreen from './LandingScreen';
import CameraPermissionScreen from './CameraPermissionScreen';
import FaceAlignmentScreen from './FaceAlignmentScreen';
import LivenessDetectionScreen from './LivenessDetectionScreen';
import SuccessScreen from './SuccessScreen';
import FailureScreen from './FailureScreen';

const App = () => (
  <Router>
    <Switch>
      <Route exact path="/" component={LandingScreen} />
      <Route path="/camera-permission" component={CameraPermissionScreen} />
      <Route path="/face-alignment" component={FaceAlignmentScreen} />
      <Route path="/liveness-detection" component={LivenessDetectionScreen} />
      <Route path="/success" component={SuccessScreen} />
      <Route path="/failure" component={FailureScreen} />
    </Switch>
  </Router>
);

export default App;
