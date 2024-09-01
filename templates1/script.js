document.getElementById('startAuthBtn').addEventListener('click', function() {
    showScreen('camera-permission');
});

document.getElementById('grantCameraBtn').addEventListener('click', function() {
    startVideoFeed();
    showScreen('face-alignment');
});

document.getElementById('cancelBtn').addEventListener('click', function() {
    stopVideoFeed();
    showScreen('landing');
});

document.getElementById('retryBtn').addEventListener('click', function() {
    startVideoFeed();
    showScreen('liveness-detection');
});

function showScreen(screenId) {
    document.querySelectorAll('.screen').forEach(screen => {
        screen.style.display = 'none';
    });
    document.querySelector(`.screen.${screenId}`).style.display = 'block';
}

function startVideoFeed() {
    const video = document.getElementById('videoFeed');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => console.error('Error accessing the camera: ', err));
}

function stopVideoFeed() {
    const video = document.getElementById('videoFeed');
    if (video.srcObject) {
        const stream = video.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
}

// Example for simulating liveness check
function simulateLivenessCheck() {
    showScreen('liveness-detection');
    const progressBar = document.querySelector('.progress-bar');
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        progressBar.style.width = progress + '%';
        if (progress >= 100) {
            clearInterval(interval);
            const success = Math.random() > 0.5; // Simulate success or failure
            showScreen(success ? 'result-success' : 'result-failure');
        }
    }, 200);
}
