import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file

app = Flask(__name__, template_folder='../frontend')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- SYSTEM DIRECTORIES ---
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "static", "known_faces")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
for folder in [KNOWN_FACES_DIR, UPLOAD_FOLDER, REPORTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- GLOBAL DATABANKS ---
known_face_encodings = []
known_face_names = []

# attendance_db = {"2026-05-10": {"Section A": {"Math": set("John", "Doe")}}}
attendance_db = {}
active_date = datetime.now().strftime("%Y-%m-%d")
active_subject = "General Entry"

# student_registry = {"Name": {"roll": "1", "section": "A", "photo": "/path"}}
student_registry = {}
master_sheet_data = [] 

def mark_attendance(name, section):
    global active_date, active_subject
    if active_date not in attendance_db: attendance_db[active_date] = {}
    if section not in attendance_db[active_date]: attendance_db[active_date][section] = {}
    if active_subject not in attendance_db[active_date][section]: attendance_db[active_date][section][active_subject] = set()
    
    attendance_db[active_date][section][active_subject].add(name)

def sync_memory_core():
    global known_face_encodings, known_face_names, student_registry
    known_face_encodings.clear()
    known_face_names.clear()
    print("\n✨ INITIALIZING NEXPRESENCE AI CORE...")
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            # Format expected: Name_Roll_Section.jpg -> handled gracefully if old format
            parts = os.path.splitext(filename)[0].split('_')
            name = parts[0]
            roll = parts[1] if len(parts) > 1 else "Unassigned"
            sec = parts[2] if len(parts) > 2 else "Unassigned"
            
            try:
                image = cv2.imread(image_path)
                if image is None: continue
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                clean_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
                locations = face_recognition.face_locations(clean_image)
                
                if locations:
                    encoding = face_recognition.face_encodings(clean_image, locations)[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                    student_registry[name] = {"roll": roll, "section": sec, "photo": f"/static/known_faces/{filename}"}
            except Exception as e:
                print(f"   [!] CORRUPTION DETECTED in {filename}: {e}")
                
    print(f"🛰️ Total Operators Synced: {len(known_face_names)}\n")

sync_memory_core()

# --- LOW-LIGHT VISION SUBSYSTEM ---
def enhance_low_light(frame):
    """Applies CLAHE to artificially enhance contrast and brighten dark faces."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def generate_frames(section="Unassigned"):
    camera = cv2.VideoCapture(1) 
    if not camera.isOpened(): camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    process_this_frame = True
    face_locations, face_names = [], []

    while True:
        success, frame = camera.read()
        if not success: break
        
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # THE FIX: Apply Low-Light Enhancer before Face Detection
            bright_frame = enhance_low_light(small_frame)
            rgb_small_frame = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        # Only mark attendance if they belong to the active teacher's section
                        student_sec = student_registry.get(name, {}).get("section", "Unassigned")
                        if student_sec.lower() == section.lower() or section == "Unassigned":
                            mark_attendance(name, student_sec)
                face_names.append(name)
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 245, 212) if name != "Unknown" else (0, 0, 255) 
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            text_color = (0, 0, 0) if name != "Unknown" else (255, 255, 255)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, text_color, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()

# --- NETWORK ROUTES ---
@app.route('/')
def index(): return render_template('indexPart.html')

@app.route('/set_config', methods=['POST'])
def set_config():
    global active_date, active_subject
    data = request.get_json()
    if 'date' in data: active_date = data['date']
    if 'subject' in data: active_subject = data['subject']
    return jsonify({"status": f"System calibrated for {active_subject} on {active_date}"})

@app.route('/video_feed')
def video_feed(): 
    section = request.args.get('section', 'Unassigned')
    return Response(generate_frames(section), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_student', methods=['POST'])
def register_student():
    name = request.form.get('name')
    roll = request.form.get('roll', 'Unassigned')
    sec = request.form.get('section', 'Unassigned')
    file = request.files.get('file')
    
    if not name or not file: return "Missing parameters", 400

    filename = f"{name.strip()}_{roll.strip()}_{sec.strip()}.jpg"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None: return "Image Matrix Unreadable", 400

        # Enhance image to make sure we extract a good encoding
        bright_image = enhance_low_light(image)
        rgb_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB)
        
        if not face_recognition.face_locations(rgb_image): 
            return "No facial signature detected. Please upload a clear photo.", 400
            
        cv2.imwrite(filepath, image)
        sync_memory_core() 
        return f"Operator {name} [Roll: {roll}, Sec: {sec}] signature registered.", 200
    except Exception as e:
        return f"System Error: {str(e)}", 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    log_path = os.path.join(REPORTS_FOLDER, 'diagnostics_log.txt')
    with open(log_path, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {data.get('name', 'Anon')}:\n{data.get('message', '')}\n{'-'*40}\n")
    return jsonify({"status": "Diagnostics safely transmitted to Core."})

@app.route('/upload_master_sheet', methods=['POST'])
def upload_master_sheet():
    global master_sheet_data
    if 'file' not in request.files: return "No file payload", 400
    file = request.files['file']
    section_filter = request.form.get('section', 'Unassigned') # Teacher's section
    
    try:
        if file.filename.endswith('.csv'): df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')): df = pd.read_excel(file)
        else: return "Unsupported file architecture", 400

        df.columns = [str(c).lower().strip().replace(' ', '') for c in df.columns]
        name_col = next((c for c in df.columns if 'name' in c), None)
        roll_col = next((c for c in df.columns if 'roll' in c), None)
        sec_col = next((c for c in df.columns if 'sec' in c or 'class' in c), None)

        if not name_col or not roll_col:
            return "Data structure invalid. Missing 'Name' or 'Roll' headers.", 400

        # Don't clear entirely, just overwrite the specific section
        master_sheet_data = [d for d in master_sheet_data if d.get('section') != section_filter]
        
        for _, row in df.iterrows():
            if pd.isna(row[name_col]): continue 
            
            sheet_name = str(row[name_col]).strip()
            roll_no = str(row[roll_col]).strip() if not pd.isna(row[roll_col]) else "N/A"
            student_sec = str(row[sec_col]).strip() if sec_col and not pd.isna(row[sec_col]) else section_filter
            
            # Map Roll to Memory
            for known_name in student_registry.keys():
                if known_name.lower() == sheet_name.lower():
                    student_registry[known_name]["roll"] = roll_no
                    student_registry[known_name]["section"] = student_sec
                    break
                    
            # Only add to UI list if it matches the teacher's section
            if student_sec.lower() == section_filter.lower():
                master_sheet_data.append({"name": sheet_name, "roll": roll_no, "section": student_sec})
            
        return f"Master Registry Synced for Section {section_filter}.", 200
    except Exception as e:
        return f"File Error: {str(e)}", 500

@app.route('/upload_classroom', methods=['POST'])
def upload_classroom():
    if 'file' not in request.files: return "No file", 400
    section = request.form.get('section', 'Unassigned')
    
    file_bytes = np.frombuffer(request.files['file'].read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    bright_image = enhance_low_light(image)
    rgb_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=2)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                sec = student_registry.get(name, {}).get("section", "Unassigned")
                if sec.lower() == section.lower():
                    mark_attendance(name, sec) 
    return "Deep Image Scan Complete.", 200

@app.route('/attendance_data')
def attendance_data():
    section = request.args.get('section', 'Unassigned')
    
    # Get students ONLY from this section
    section_students = [n for n, info in student_registry.items() if info.get('section', '').lower() == section.lower()]
    
    # Get attendance for active date, active section, active subject
    present_set = attendance_db.get(active_date, {}).get(section, {}).get(active_subject, set())
    
    present = list(present_set)
    absent = [s for s in section_students if s not in present]
    
    return jsonify({"total": len(section_students), "present": present, "absent": absent})

@app.route('/dashboard_analytics')
def dashboard_analytics():
    section = request.args.get('section', 'Unassigned')
    
    # Filter registry for just this section
    filtered_registry = {n: i for n, i in student_registry.items() if i.get('section', '').lower() == section.lower()}
    
    # Total days this section had ANY class
    section_history = {} # {date: [names present]}
    for date, secs in attendance_db.items():
        if section in secs:
            # Union of all subjects for the day
            daily_present = set()
            for sub, names in secs[section].items():
                daily_present.update(names)
            section_history[date] = list(daily_present)
            
    total_days = len(section_history)
    daily_stats = {d: len(s) for d, s in section_history.items()}
    
    streak_holders = [s for s in filtered_registry.keys() if all(s in section_history[d] for d in section_history)] if total_days > 0 else []

    individual_reports = []
    for name, info in filtered_registry.items():
        present_days = sum(1 for date, attendees in section_history.items() if name in attendees)
        percentage = (present_days / total_days * 100) if total_days > 0 else 0
        status = "EXCELLENT" if percentage >= 90 else "GOOD" if percentage >= 75 else "AVERAGE" if percentage >= 50 else "POOR"
        history = {date: (1 if name in attendees else 0) for date, attendees in section_history.items()}
        individual_reports.append({
            "name": name, "roll": info["roll"], "photo": info["photo"],
            "percentage": round(percentage, 1), "status": status, "history": history
        })

    roster_status = []
    for sheet_student in master_sheet_data:
        if sheet_student.get("section", "").lower() == section.lower():
            is_enrolled = any(known_name.lower() == sheet_student["name"].lower() for known_name in known_face_names)
            roster_status.append({"name": sheet_student["name"], "roll": sheet_student["roll"], "enrolled": is_enrolled})

    return jsonify({
        "total_enrolled": len(filtered_registry), "total_days": total_days,
        "daily_stats": daily_stats, "streak_holders": streak_holders,
        "individual_reports": individual_reports, "roster_status": roster_status
    })

@app.route('/export_csv')
def export_csv():
    section = request.args.get('section', 'Unassigned')
    filepath = os.path.join(REPORTS_FOLDER, f"NexPresence_{section}_{active_date}.csv")
    
    section_students = [n for n, info in student_registry.items() if info.get('section', '').lower() == section.lower()]
    present_set = attendance_db.get(active_date, {}).get(section, {}).get(active_subject, set())
    
    present = list(present_set)
    absent = [s for s in section_students if s not in present]
    
    with open(filepath, 'w') as f:
        f.write("Roll Number,Operator Name,Status,Log Date,Subject\n")
        for name in present: f.write(f"{student_registry.get(name, {}).get('roll', 'N/A')},{name},Present,{active_date},{active_subject}\n")
        for name in absent: f.write(f"{student_registry.get(name, {}).get('roll', 'N/A')},{name},Absent,{active_date},{active_subject}\n")
    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)