from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from model_utils import detector
from datetime import datetime
import sqlite3

app = Flask(__name__)
app.static_folder = 'static'

# Initialize database
def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT,
                  result TEXT,
                  description TEXT,
                  confidence REAL,
                  class TEXT,
                  latitude REAL,
                  longitude REAL,
                  district TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# Uganda's approximate bounds
UGANDA_LAT_MIN = -1.5
UGANDA_LAT_MAX = 4.2
UGANDA_LON_MIN = 29.5
UGANDA_LON_MAX = 35.0

# List of Uganda districts for reference
UGANDA_DISTRICTS = [
    "Kampala", "Wakiso", "Mukono", "Jinja", "Mbale", "Mbarara", "Gulu", "Lira",
    "Arua", "Masaka", "Kabale", "Fort Portal", "Hoima", "Soroti", "Kampala"
]

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Get location data from request
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)
        district = request.form.get('district', type=str)
        
        # Check if location data is missing or invalid
        location_missing = (latitude is None or longitude is None or
                           not (UGANDA_LAT_MIN <= latitude <= UGANDA_LAT_MAX) or
                           not (UGANDA_LON_MIN <= longitude <= UGANDA_LON_MAX))
        
        # Return error if location data is missing or invalid
        if location_missing:
            return jsonify({
                "error": "Valid location data is required. Please provide latitude and longitude within Uganda's boundaries.",
                "bounds": {
                    "lat_min": UGANDA_LAT_MIN,
                    "lat_max": UGANDA_LAT_MAX,
                    "lon_min": UGANDA_LON_MIN,
                    "lon_max": UGANDA_LON_MAX
                }
            }), 400
        
        # If district is not provided, set it to empty string
        if district is None:
            district = ""
        
        # Create uploads directory if it doesn't exist
        os.makedirs('static/uploads', exist_ok=True)
        
        # Save the file with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        image_path = os.path.join('static/uploads', filename)
        file.save(image_path)
        
        # Run detection
        results = detector.detect(image_path)
        
        # Map the result to the correct class
        detection_class = 'unknown'
        if 'result' in results:
            result_text = results['result'].lower()
            if 'larval damage' in result_text:
                detection_class = 'fall-armyworm-larval-damage'
            elif 'egg' in result_text:
                detection_class = 'fall-armyworm-egg'
            elif 'frass' in result_text:
                detection_class = 'fall-armyworm-frass'
            elif 'healthy' in result_text:
                detection_class = 'healthy-maize'
        
        # Add the class to results
        results['class'] = detection_class
        
        # Store results in database if it's a maize leaf
        if results.get('is_maize', False):
            conn = sqlite3.connect('detections.db')
            c = conn.cursor()
            c.execute('''INSERT INTO detections
                         (image_path, result, description, confidence, class, latitude, longitude, district)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (image_path, results['result'], results['description'],
                      results['confidence'], detection_class,
                      latitude, longitude, district))
            conn.commit()
            
            # Get the ID of the inserted record
            detection_id = c.lastrowid
            conn.close()
            
            # Add location and ID to results
            results['latitude'] = latitude
            results['longitude'] = longitude
            results['district'] = district
            results['id'] = detection_id
            results['image_path'] = image_path
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_location/<int:detection_id>', methods=['POST'])
def update_location(detection_id):
    """Endpoint to update location data for a detection"""
    try:
        # Get updated location data
        latitude = request.json.get('latitude', type=float)
        longitude = request.json.get('longitude', type=float)
        district = request.json.get('district')
        
        # Validate location data
        if not latitude or not longitude or not district:
            return jsonify({"error": "Missing location data"}), 400
        
        if not (UGANDA_LAT_MIN <= latitude <= UGANDA_LAT_MAX) or not (UGANDA_LON_MIN <= longitude <= UGANDA_LON_MAX):
            return jsonify({"error": "Coordinates outside Uganda"}), 400
        
        # Update the database
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute('''UPDATE detections
                     SET latitude = ?, longitude = ?, district = ?
                     WHERE id = ?''',
                 (latitude, longitude, district, detection_id))
        conn.commit()
        
        if c.rowcount == 0:
            conn.close()
            return jsonify({"error": "Detection not found"}), 404
        
        conn.close()
        return jsonify({"success": True, "message": "Location updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/map_data', methods=['GET'])
def get_map_data():
    try:
        # Get parameters for filtering (optional)
        days = request.args.get('days', default=30, type=int)
        class_filter = request.args.get('class', default=None, type=str)
        district_filter = request.args.get('district', default=None, type=str)
        
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        
        # Base query with time filter
        query = '''SELECT latitude, longitude, class, result, confidence, district, timestamp, id, image_path
                   FROM detections
                   WHERE timestamp >= datetime('now', ?)'''
        params = [f'-{days} days']
        
        # Add Uganda bounds filter
        query += ''' AND latitude BETWEEN ? AND ?
                     AND longitude BETWEEN ? AND ?'''
        params.extend([UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX])
        
        # Add optional class filter
        if class_filter:
            query += ' AND class = ?'
            params.append(class_filter)
        
        # Add optional district filter
        if district_filter:
            query += ' AND district = ?'
            params.append(district_filter)
        
        c.execute(query, params)
        detections = c.fetchall()
        conn.close()
        
        # Format the data for the map
        map_data = []
        for detection in detections:
            map_data.append({
                'latitude': detection[0],
                'longitude': detection[1],
                'class': detection[2],
                'result': detection[3],
                'confidence': detection[4],
                'district': detection[5],
                'timestamp': detection[6],
                'id': detection[7],
                'image_path': detection[8]
            })
        
        return jsonify(map_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/map_view')
def map_view():
    return render_template('index.html')

@app.route('/uganda_districts', methods=['GET'])
def uganda_districts():
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        
        # Get unique districts within Uganda's bounds
        query = '''SELECT DISTINCT district
                   FROM detections
                   WHERE latitude BETWEEN ? AND ?
                   AND longitude BETWEEN ? AND ?
                   AND district IS NOT NULL
                   ORDER BY district'''
        
        c.execute(query, [UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX])
        districts = c.fetchall()
        conn.close()
        
        # If no districts in database yet, return the default list
        if not districts:
            return jsonify(UGANDA_DISTRICTS)
        
        return jsonify([district[0] for district in districts])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
