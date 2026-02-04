from flask import Flask, render_template, flash, redirect, request, send_from_directory, url_for, send_file, jsonify, Response
from PIL import Image
import numpy as np
from tensorflow import keras
import cv2
import pymysql
import os
import threading
import time
from datetime import datetime
import pyautogui
import mss
import mss.tools
from queue import Queue
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="root",
    port=3306,
    database='video_detection'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('login.html', message="This email ID is already exists!")
        return render_template('login.html', message="Conform password is not match!")
    return render_template('login.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('home.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')




### Fake video detection


MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
IMG_SIZE = 224

# Load the model
model = keras.models.load_model(r"Models\model.h5")

# Define the feature extractor
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Function to prepare a single video for prediction
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


# Utility function to load video frames
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# Utility function to crop center square of a frame
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


@app.route('/video_pred', methods=['GET', 'POST'])
def video_pred():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('static/saved_videos/', fn)
        myfile.save(mypath)

        # Load the uploaded video
        frames = load_video(mypath)

        # Prepare video frames for prediction
        frame_features, frame_mask = prepare_single_video(frames)

        # Perform prediction
        prediction = model.predict([frame_features, frame_mask])[0]
        print(prediction)

        # Determine the predicted class
        if prediction >= 0.8:
            predicted_class = 'FAKE'
        else:
            predicted_class = 'REAL'
        # Pass the prediction result to the template
        return render_template('video.html', path=mypath, prediction=predicted_class)
    return render_template('video.html')






### Fake Image Prediction

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


@app.route('/image_pred', methods=['GET', 'POST'])
def image_pred():
    if request.method == 'POST':
        classes = ["Fake", "Real"]

        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join('static/saved_image/', fn)
        myfile.save(mypath)

        models = load_model(r"Models\cnn.h5")
        x = image.load_img(mypath, target_size=(256, 256))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x /= 255
        results = models.predict(x)
        b = np.argmax(results)
        prediction = classes[b]

        return render_template('image.html', path=mypath, prediction = prediction)
    return render_template('image.html')


is_recording = False
recording_thread = None
recording_start_time = None
current_recording_path = None
recording_lock = threading.Lock()

### Screen Recording Functions ###
def screen_recorder():
    global is_recording, recording_start_time, current_recording_path
    
    # Create screenshots folder if it doesn't exist
    recordings_dir = 'static/screen_recordings'
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screen_recording_{timestamp}.mp4"
    output_path = os.path.join(recordings_dir, filename)
    
    with recording_lock:
        current_recording_path = output_path
        recording_start_time = time.time()
    
    print(f"Recording started: {output_path}")
    
    try:
        # Get screen size using mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            width = monitor["width"]
            height = monitor["height"]
            
            # Define the codec and create VideoWriter object
            # Using mp4v codec for MP4 format
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 15.0  # Reduced FPS for better performance
            
            # Create video writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print(f"Screen resolution: {width}x{height}, FPS: {fps}")
            
            frame_count = 0
            start_time = time.time()
            
            while is_recording:
                # Capture screen
                screenshot = sct.grab(monitor)
                
                # Convert to numpy array
                img = np.array(screenshot)
                
                # Convert BGRA to BGR (remove alpha channel)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Write frame
                out.write(img)
                frame_count += 1
                
                # Calculate actual frame rate
                elapsed_time = time.time() - start_time
                actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Adjust sleep time to maintain desired FPS
                target_frame_time = 1.0 / fps
                processing_time = time.time() - start_time - elapsed_time
                sleep_time = max(0, target_frame_time - processing_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Break if recording is stopped
                if not is_recording:
                    break
            
            # Release resources
            out.release()
            cv2.destroyAllWindows()
            
            recording_duration = time.time() - start_time
            print(f"Recording completed: {output_path}")
            print(f"Duration: {recording_duration:.2f} seconds")
            print(f"Total frames: {frame_count}")
            print(f"Average FPS: {frame_count / recording_duration if recording_duration > 0 else 0:.2f}")
            
    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        # Cleanup
        with recording_lock:
            if is_recording:  # If still recording, something went wrong
                is_recording = False
    
    return output_path

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_thread
    
    if not is_recording:
        is_recording = True
        
        # Start recording in a separate thread
        recording_thread = threading.Thread(target=screen_recorder)
        recording_thread.daemon = True
        recording_thread.start()
        
        # Wait a moment to ensure recording has started
        time.sleep(0.5)
        
        return jsonify({
            'status': 'success', 
            'message': 'Screen recording started',
            'is_recording': True
        })
    
    return jsonify({
        'status': 'error', 
        'message': 'Recording is already in progress',
        'is_recording': True
    })

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording, recording_start_time, current_recording_path
    
    if is_recording:
        # Stop recording
        is_recording = False
        
        # Wait for recording thread to finish
        if recording_thread and recording_thread.is_alive():
            recording_thread.join(timeout=5)
        
        # Calculate recording duration
        duration = 0
        if recording_start_time:
            duration = time.time() - recording_start_time
        
        # Reset start time
        recording_start_time = None
        
        # Check if file was created
        saved_path = None
        file_size = 0
        if current_recording_path and os.path.exists(current_recording_path):
            saved_path = current_recording_path
            file_size = os.path.getsize(saved_path) / (1024 * 1024)  # Size in MB
        
        return jsonify({
            'status': 'success', 
            'message': 'Screen recording stopped',
            'file_path': f'/{saved_path}' if saved_path else None,
            'duration': f"{duration:.1f} seconds",
            'file_size': f"{file_size:.2f} MB",
            'is_recording': False
        })
    
    return jsonify({
        'status': 'error', 
        'message': 'No recording in progress',
        'is_recording': False
    })

@app.route('/recording_status', methods=['GET'])
def recording_status():
    global is_recording, recording_start_time
    
    duration = 0
    if is_recording and recording_start_time:
        duration = time.time() - recording_start_time
    
    return jsonify({
        'is_recording': is_recording,
        'duration': f"{duration:.1f}"
    })

@app.route('/get_recordings', methods=['GET'])
def get_recordings():
    recordings = []
    recordings_folder = 'static/screen_recordings'
    
    if os.path.exists(recordings_folder):
        # Get all video files
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        files = [f for f in os.listdir(recordings_folder) 
                if f.lower().endswith(video_extensions)]
        
        for file in files:
            file_path = os.path.join(recordings_folder, file)
            relative_path = f'/static/screen_recordings/{file}'
            file_time = os.path.getmtime(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            formatted_time = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
            recordings.append({
                'name': file,
                'path': relative_path,
                'time': formatted_time,
                'size': f"{file_size:.2f} MB"
            })
    
    # Sort by time (newest first)
    recordings.sort(key=lambda x: x['time'], reverse=True)
    
    return jsonify({'recordings': recordings})

# DELETE RECORDING ENDPOINT - Make sure this is added
@app.route('/delete_recording', methods=['POST'])
def delete_recording():
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'status': 'error', 'message': 'No filename provided'}), 400
        
        # Security check: ensure the filename doesn't contain path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'status': 'error', 'message': 'Invalid filename'}), 400
        
        recordings_folder = 'static/screen_recordings'
        file_path = os.path.join(recordings_folder, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted recording: {file_path}")
            return jsonify({'status': 'success', 'message': 'Recording deleted successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
            
    except Exception as e:
        print(f"Error deleting recording: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Ensure the recordings directory exists
    if not os.path.exists('static/screen_recordings'):
        os.makedirs('static/screen_recordings')
    
    # Clean up any zombie threads on startup
    is_recording = False
    
    app.run(debug=True, threaded=True, use_reloader=False)