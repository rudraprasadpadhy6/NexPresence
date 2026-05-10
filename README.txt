# NexPresence AI Core 🌌

NexPresence AI Core is a modern, lightweight, face-recognition-powered smart attendance system. It features a robust Python Flask backend, real-time computer vision capabilities, and a sleek, deep-space-themed web interface for seamless monitoring and reporting.

## 🚀 Key Features

*   **Real-time Facial Recognition:** Utilizes OpenCV and the `face_recognition` library to detect and match faces via a live camera feed.
*   **Low-Light Vision Module:** Automatically applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to brighten and enhance faces in poorly lit environments before scanning.
*   **Master Roster Sync:** Upload class lists via `.csv` or `.xlsx` files to map student names, roll numbers, and sections to their biometric profiles.
*   **Dynamic Analytics Dashboard:** Visualizes attendance regularity, individual operator logs, streaks, and overall class statistics.
*   **Photo Wash Utility:** Includes `WashPhotos.py` to standardize uploaded images, strip bad metadata, and convert them to pure `.jpg` files for the AI engine.
*   **CSV Data Export:** Generate and export detailed daily attendance logs into easy-to-read CSV files.

## 📁 Project Structure

The project has been organized into a clean `frontend` and `backend` architecture.

```text
NexPresence/
│
├── backend/
│   ├── WebApp.py            # Main Flask application and AI logic
│   ├── WashPhotos.py        # Utility script to clean and format images
│   ├── static/
│   │   └── known_faces/     # Database of enrolled operator facial signatures
│   ├── uploads/             # Temporary folder for file uploads
│   └── reports/             # Generated CSV reports and diagnostic logs
│
└── frontend/
    └── indexPart.html       # The main UI (HTML/CSS/JS embedded)
```

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   A C++ Compiler (Required for `dlib` installation on Windows)
*   A connected webcam

### 1. Install Dependencies
Make sure you are inside your virtual environment (if you are using one, such as Anaconda or venv). Install the required Python packages:

```bash
pip install flask opencv-python face_recognition numpy pandas pillow
```
*(Note: `face_recognition` requires `dlib`. If you encounter errors on Windows, you may need to install CMake and Visual Studio C++ Build Tools first).*

### 2. Running the System
Navigate to the `backend` directory and start the Flask server:

```bash
cd backend
python WebApp.py
```

### 3. Accessing the Dashboard
Open your web browser and navigate to:
**http://localhost:5000**

## 📖 How to Use

1. **Authorization:** Log in as an Administrator (Teacher) or Operator (Student). Select your sector and provide your credentials.
2. **Schedule Matrix:** Navigate to the "Schedule & Config" tab to set the active subject and date for the session.
3. **Sync Roster:** Go to the "Scanner Engine" tab and upload your `.csv` or `.xlsx` Master Sheet to populate the student registry for your sector.
4. **Enroll Operators:** Use the "Enroll Operator Profile" section on the Dashboard to upload a clear portrait of each student.
5. **Engage Scanner:** In the "Scanner Engine" tab, click "Engage Camera" to start the live feed. The system will automatically mark operators as present when they appear on camera.
6. **Export Data:** Click "Export CSV" to download the finalized attendance sheet for the day.

## 🧹 Maintenance (WashPhotos.py)
If you encounter issues with users uploading images from smartphones with weird color profiles or formats (like transparent PNGs), you can run the digital wash tool:

```bash
cd backend
python WashPhotos.py
```
This will strip metadata and force all images in the `known_faces` folder to a standardized RGB `.jpg` format, drastically improving AI recognition accuracy.
