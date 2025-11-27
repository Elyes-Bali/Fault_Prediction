import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import io
import time # <--- ADDED: Import time module for performance measurement

# --- FLASK SETUP ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- MODEL PATH (MUST BE ACCESSIBLE TO THE SERVER) ---
# NOTE: Replace this with the actual path on your server if different
MODEL_PATH =r"D:\Ayoub\kerasCode\modele_detection_panne.keras" 

# --- MODEL PARAMETERS (From your test2.ipynb) ---
BATCH_SIZE = 4096 
WINDOW_SIZE = 1000
BEST_THRESHOLD = 0.69

FEATURE_COLUMNS = [
  'FIC2003_valeur', 'LI2001_valeur', 'MX2001_valeur',
  'MX2002A_valeur', 'MX2002B_valeur', 'MX2002C_valeur',
  'MX2002D_valeur', 'MX2003A_valeur', 'MX2003B_valeur',
  'MX2003C_valeur', 'temperature_ext', 'TI2001__2_valeur'
]

# --- 2. OPTIMIZED GENERATOR FUNCTION ---
def sequence_generator(data, time_steps, batch_size):
  """Yields batches of scaled sequences, returning a tuple as required by Keras."""
  n_sequences = len(data) - time_steps + 1
  for i in range(0, n_sequences, batch_size):
    batch_end_index = min(i + batch_size, n_sequences)
    X_batch = []
    for seq_index in range(i, batch_end_index):
      X_batch.append(data[seq_index : (seq_index + time_steps)])
    yield (np.array(X_batch),) 

# --- 3. CORE INFERENCE FUNCTION ---
def run_inference(csv_path):
  # Redirect standard output to capture print logs
  old_stdout = sys.stdout
  sys.stdout = log_stream = io.StringIO()
  
  try:
    print(f"Loading data from: {csv_path}")
    
    # Load Data
    df_raw = pd.read_csv(csv_path)
    
    # ⚠️ CAUSE IDENTIFIED: This line limits the data to only 99,000 rows.
    # If the original file was much larger, this is why the API call is fast.
    # df_raw=df_raw.head(99000)
    
    if 'Unnamed: 0' in df_raw.columns:
      df_raw = df_raw.drop('Unnamed: 0', axis=1)

    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_features = df_raw[FEATURE_COLUMNS].copy()
    data_to_process = df_features.values
    
    print(f"Data shape: {df_raw.shape}. Features extracted: {df_features.shape[1]}")

    # --- TEMPORARY NORMALIZATION STEP ---
    # ⚠️ WARNING: Still using INACCURATE statistics derived from the new data.
    print("Applying temporary StandardScaler (Warning: for accurate results, use TRAINING statistics).")
    temp_scaler = StandardScaler()
    data_scaled = temp_scaler.fit_transform(data_to_process)

    # Load the trained Keras model
    print(f"Loading Keras model from: {MODEL_PATH}")
    # Model loading time starts here
    model_load_start = time.time()
    model = keras.models.load_model(MODEL_PATH)
    model_load_end = time.time()
    print(f"Model loading time: {model_load_end - model_load_start:.2f} seconds.") # <--- ADDED LOG
    
    n_sequences = len(data_scaled) - WINDOW_SIZE + 1
    if n_sequences <= 0:
      raise ValueError(f"Data too short ({len(data_scaled)} rows) for WINDOW_SIZE {WINDOW_SIZE}.")
      
    steps_per_epoch = int(np.ceil(n_sequences / BATCH_SIZE))
    
    # Prediction using the generator
    print(f"Starting prediction across {n_sequences} sequences in {steps_per_epoch} batches...")
    
    # --- START INFERENCE TIMER ---
    inference_start_time = time.time()
    
    y_proba = model.predict(
      sequence_generator(data_scaled, WINDOW_SIZE, BATCH_SIZE),
      steps=steps_per_epoch,
      verbose=1 
    ).flatten()

    inference_end_time = time.time()
    # --- END INFERENCE TIMER ---
    print(f"Prediction complete. Total inference time: {inference_end_time - inference_start_time:.2f} seconds.") # <--- ADDED LOG

    # Interpretation
    y_pred_binary = (y_proba >= BEST_THRESHOLD).astype(int)
    prediction_labels = np.where(y_pred_binary == 1, 'Fault (Panne)', 'Normal')

    # Create Initial Results Table
    start_index = WINDOW_SIZE - 1
    datetime_end = df_raw['datetime'].iloc[start_index:].reset_index(drop=True)

    results_df = pd.DataFrame({
      'TimeWindow_Ends_At': datetime_end.iloc[:len(y_proba)],
      'Fault_Probability': y_proba,
      'Prediction': prediction_labels
    })

    # ----------------------------------------------------
    # 4. FAULT EVENT ANALYSIS LOGIC
    # ----------------------------------------------------
    
    print("Analyzing fault events...")
    # Calculate the time step duration (assuming uniform sampling)
    time_step_duration = df_raw['datetime'].iloc[1] - df_raw['datetime'].iloc[0]

    # Calculate Start Time and Reorder columns
    time_offset = (WINDOW_SIZE - 1) * time_step_duration
    results_df['TimeWindow_Start_At'] = results_df['TimeWindow_Ends_At'] - time_offset
    results_df = results_df[['TimeWindow_Start_At', 'TimeWindow_Ends_At', 'Fault_Probability', 'Prediction']]

    # Identify fault predictions
    is_fault = results_df['Prediction'] == 'Fault (Panne)'
    group_change = is_fault.diff().fillna(is_fault.iloc[0]) 
    group_id = (group_change == True).cumsum()

    # Aggregate to find the start and end of each continuous fault event
    fault_events = results_df[is_fault].copy()
    if not fault_events.empty:
      fault_events['group'] = group_id[is_fault]
      fault_summary = fault_events.groupby('group').agg(
        Fault_Start_Time=('TimeWindow_Start_At', 'min'), 
        Fault_End_Time=('TimeWindow_Ends_At', 'max'),   
        Max_Probability=('Fault_Probability', 'max'),
        Duration_Windows=('Prediction', 'count')
      ).reset_index(drop=True)
      fault_summary['Duration'] = fault_summary['Fault_End_Time'] - fault_summary['Fault_Start_Time']
    else:
      fault_summary = pd.DataFrame(columns=['Fault_Start_Time', 'Fault_End_Time', 'Max_Probability', 'Duration_Windows', 'Duration'])
    
    print("\nInference Complete. Results generated.")

    # Convert results to JSON format suitable for React table display
    results = {
      'logs': log_stream.getvalue(),
      'summary_data': fault_summary.to_json(orient='split', date_format='iso'),
      'detailed_data': results_df.head(100).to_json(orient='split', date_format='iso'), # Limit detailed data to 100 rows for display
      'status': 'success'
    }
    return results

  except Exception as e:
    # ... (error handling) ...
    print(f"An error occurred: {str(e)}")
    return {
      'logs': log_stream.getvalue(),
      'status': 'error',
      'error_message': str(e)
    }
  finally:
    # Restore standard output
    sys.stdout = old_stdout


# --- API ENDPOINTS ---

@app.route('/api/upload-preview', methods=['POST'])
def upload_preview():
  """Handles CSV upload and returns table preview data."""
  # ... (function body unchanged) ...
  if 'file' not in request.files:
    return jsonify({'error': 'No file part'}), 400
  file = request.files['file']
  if file.filename == '':
    return jsonify({'error': 'No selected file'}), 400
  if file:
    try:
      # Read CSV content directly into pandas DataFrame for preview
      df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
      
      # Use only the first 10 rows for a quick preview
    #   preview_df = df.head(40).fillna('N/A')
      preview_df = df.fillna('N/A')
      
      return jsonify({
        'status': 'preview_ready',
        'filename': file.filename,
        'preview_data': preview_df.to_json(orient='split', date_format='iso')
      })
    except Exception as e:
      return jsonify({'error': f"Failed to read CSV: {str(e)}"}), 500


@app.route('/api/run-model', methods=['POST'])
def run_model():
  """Handles the final file upload and runs the ML inference."""
  # ... (function body unchanged) ...
  if 'file' not in request.files:
    return jsonify({'error': 'No file part'}), 400
  file = request.files['file']
  
  filename = secure_filename(file.filename)
  filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  
  # Save the file temporarily
  file.save(filepath)
  
  # Run the core ML inference
  results = run_inference(filepath)
  
  # Clean up the temporary file
  os.remove(filepath)
  
  return jsonify(results)

if __name__ == '__main__':
  # You must run this command in your terminal:
  # pip install flask tensorflow scikit-learn pandas werkzeug
  #
  # Ensure your Keras model is in the same directory.
  # To run: python app.py
  app.run(debug=True, port=5000)