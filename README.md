
<center>  
# Face Liveness Detection (Anti-Spoofing) Web App  
A **Streamlit WebRTC** app that detects whether a face on camera is real or spoofed, preventing unauthorized access and enhancing system security.  
</center>  

---

## 🌟 Inspiration  
This project was inspired by [jomariya23156's face recognition web app](https://github.com/jomariya23156/face-recognition-with-liveness-web-login). While an excellent starting point, the app lacks comprehensive WebRTC functionality needed for a server-client environment. For additional context on building WebRTC with Streamlit, check out [this blog post](https://blog.streamlit.io/how-to-build-the-streamlit-webrtc-component/).

---

## 🚀 App Features  
- **Liveness Detection**: Calculates real vs. fake face detection ratios.
- **WebRTC Integration**: Utilizes Streamlit’s WebRTC features for real-time detection.

---

## 🏁 Quick Start  

### Step 1: Clone the Repository  
Fork and clone the repo locally:

```sh
git clone https://github.com/shivampratap164/Face-Liveness-Detection-Anti-Spoofing-Web-App.git
```

### Step 2: Create and Activate a Virtual Environment  
To keep dependencies organized, create and activate a virtual environment:

```sh
pip install virtualenv
python -m venv [env-name]
source [env-name]/bin/activate  # For MacOS/Linux
[env-name]\Scripts\activate     # For Windows
```

### Step 3: Navigate to Project Directory  

```sh
cd Face-Liveness-Detection-Anti-Spoofing-Web-App
```

### Step 4: Install Dependencies  

```sh
pip install -r requirements.txt
```

### Step 5: Run the App  

```sh
streamlit run app.py
```

The application should now be accessible at [http://localhost:8501](http://localhost:8501).

### Step 6: Deploy to the Cloud (Optional)  
Deploy your app to cloud platforms like Streamlit-sharing or Heroku.

**Resources for deployment:**
- [Deploying to Heroku](https://blog.jcharistech.com/2019/10/24/how-to-deploy-your-streamlit-apps-to-heroku)
- [Deploying to Streamlit Sharing](https://towardsdatascience.com/deploying-a-basic-streamlit-app-ceadae286fd0)

If you face issues deploying the app, refer to the [Streamlit remote deployment guide](https://docs.streamlit.io/knowledge-base/deploy/remote-start).

---

## 📸 Sample Output  

### Normal Detection  
![Normal](/test_pics/normal.png?raw=true "Normal")

### Detection with Spoofed Picture  
![With Picture](/test_pics/with_pic.jpeg?raw=true "With picture")

### Detection with Spoofed Video  
![With Video](/test_pics/with_video.jpeg?raw=true "With video")

---

## 🤝 Contributing  

1. Clone the repository and set up your local environment as described above.
2. Push your changes to your GitHub fork and submit a pull request.

### Pushing Your Changes  

```bash
git add .
git commit -m "feat: added new functionality"
git push YOUR_REPO_URL develop
```

---

## ⚠️ Project Limitations  

- **Limited Device Usage**: Supports 3-4 devices at a time.
- **Lighting Sensitivity**: Performance degrades in bright backgrounds.
- **Browser Variability**: Works best on Chrome and Firefox; slower on Edge.

---

**Thank you for checking out the project!**
