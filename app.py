from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from model_utils import detector
from datetime import datetime, timedelta
import sqlite3
import math
from typing import Dict, Tuple, Optional

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

# Define approximate coordinates for Uganda districts
# Format: {district_name: (latitude, longitude)}
UGANDA_DISTRICT_COORDS: Dict[str, Tuple[float, float]] = {
    "Kampala": (0.3476, 32.5825),
    "Wakiso": (0.4033, 32.4617),
    "Mukono": (0.3533, 32.7550),
    "Jinja": (0.4250, 33.2033),
    "Mbale": (1.0750, 34.1750),
    "Mbarara": (-0.6167, 30.6500),
    "Gulu": (2.7833, 32.2833),
    "Lira": (2.2500, 32.9000),
    "Arua": (3.0167, 30.9000),
    "Masaka": (-0.3333, 31.7333),
    "Kabale": (-1.2500, 30.0000),
    "Fort Portal": (0.6667, 30.2667),
    "Hoima": (1.4333, 31.3500),
    "Soroti": (1.7167, 33.6167),
    "Tororo": (0.7083, 34.1750),
    "Moroto": (2.5333, 34.6667),
    "Kitgum": (3.2833, 32.8833),
    "Kasese": (0.1833, 30.0833),
    "Entebbe": (0.0500, 32.4633),
    "Iganga": (0.6167, 33.4833),
    "Mityana": (0.4167, 32.0333),
    "Luwero": (0.8500, 32.4667),
    "Masindi": (1.6750, 31.7150),
    "Busia": (0.4667, 34.0833),
    "Mubende": (0.5833, 31.3667),
    "Ntungamo": (-0.8833, 30.2667),
    "Adjumani": (3.3667, 31.7833),
    "Moyo": (3.6500, 31.7167),
    "Nebbi": (2.4833, 31.0833),
    "Apac": (1.9833, 32.5333),
    "Kotido": (2.9833, 34.1333),
    "Rukungiri": (-0.8417, 29.9417),
    "Bushenyi": (-0.5333, 30.2167),
    "Kiboga": (0.9167, 31.7667),
    "Kibale": (0.8000, 30.7000),
    "Kabarole": (0.6500, 30.2500),
    "Kamuli": (0.9467, 33.1200),
    "Pallisa": (1.1450, 33.7100),
    "Kumi": (1.4600, 33.9367),
    "Rakai": (-0.7167, 31.5333)
}

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def get_district_from_coordinates(latitude: float, longitude: float) -> Optional[str]:
    """
    Determine the closest district based on coordinates
    """
    if not (UGANDA_LAT_MIN <= latitude <= UGANDA_LAT_MAX and 
            UGANDA_LON_MIN <= longitude <= UGANDA_LON_MAX):
        return None
    
    closest_district = None
    min_distance = float('inf')
    
    for district, coords in UGANDA_DISTRICT_COORDS.items():
        dist_lat, dist_lon = coords
        distance = calculate_distance(latitude, longitude, dist_lat, dist_lon)
        
        if distance < min_distance:
            min_distance = distance
            closest_district = district
    
    # Only return the district if it's within a reasonable distance (e.g., 50km)
    # This helps prevent assigning districts when coordinates are far from any known point
    if min_distance <= 50:
        return closest_district
    return None

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    try:
        # Get location data from request - fix parsing of form data
        latitude = None
        longitude = None
        district = None
        
        # Try to get latitude as float
        if 'latitude' in request.form:
            try:
                latitude = float(request.form['latitude'])
            except (ValueError, TypeError):
                pass
                
        # Try to get longitude as float
        if 'longitude' in request.form:
            try:
                longitude = float(request.form['longitude'])
            except (ValueError, TypeError):
                pass
                
        # Get district from form or determine from coordinates
        district = request.form.get('district', '')
        
        # If coordinates are valid but district is empty, try to determine it
        if latitude is not None and longitude is not None and not district:
            determined_district = get_district_from_coordinates(latitude, longitude)
            if determined_district:
                district = determined_district
                
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

@app.route('/debug_form_data', methods=['POST'])
def debug_form_data():
    """Debug endpoint to see what data is being received in a POST request"""
    try:
        # Get all form data
        form_data = {}
        for key in request.form:
            form_data[key] = request.form[key]
            
        # Get all files
        files = {}
        for key in request.files:
            files[key] = request.files[key].filename
            
        # Return all data for debugging
        return jsonify({
            "form_data": form_data,
            "files": files,
            "content_type": request.content_type,
            "headers": dict(request.headers)
        })
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
        if not latitude or not longitude:
            return jsonify({"error": "Missing location data"}), 400
                
        if not (UGANDA_LAT_MIN <= latitude <= UGANDA_LAT_MAX) or not (UGANDA_LON_MIN <= longitude <= UGANDA_LON_MAX):
            return jsonify({"error": "Coordinates outside Uganda"}), 400
        
        # If district is not provided, try to determine it from coordinates
        if not district:
            district = get_district_from_coordinates(latitude, longitude)
            if not district:
                district = ""  # Set to empty string if we couldn't determine district
                
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
        return jsonify({"success": True, "message": "Location updated successfully", "district": district})
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

@app.route('/api/analytics_data', methods=['GET'])
def api_analytics_data():
    """API endpoint to provide data for the analytics dashboard in the Flutter app"""
    try:
        # Get parameters for filtering
        days = request.args.get('days', default=30, type=int)
        class_filter = request.args.get('class', default=None, type=str)
        district_filter = request.args.get('district', default=None, type=str)
                
        # Connect to database
        conn = sqlite3.connect('detections.db')
        conn.row_factory = sqlite3.Row  # This enables column access by name
        c = conn.cursor()
            # Base query with time filter
        base_query = '''SELECT id, class, result, confidence, district, timestamp, latitude, longitude
                       FROM detections
                       WHERE timestamp >= datetime('now', ?)
                       AND latitude BETWEEN ? AND ?
                       AND longitude BETWEEN ? AND ?'''
        base_params = [f'-{days} days', UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX]
                
        # Add optional filters
        if class_filter:
            base_query += ' AND class = ?'
            base_params.append(class_filter)
                
        if district_filter:
            base_query += ' AND district = ?'
            base_params.append(district_filter)
                
        # Execute query to get all filtered detections
        c.execute(base_query, base_params)
        detections = [dict(row) for row in c.fetchall()]
                
        # Calculate total detections
        total_detections = len(detections)
                
        # Calculate districts affected
        districts_affected = len(set(d['district'] for d in detections if d['district']))
                
        # Calculate class distribution
        class_distribution = {}
        for detection in detections:
            detection_class = detection['class'] or 'unknown'
            class_distribution[detection_class] = class_distribution.get(detection_class, 0) + 1
                
        # Calculate infestation rate (excluding healthy maize)
        total_classified = sum(class_distribution.values())
        infestation_count = total_classified - class_distribution.get('healthy-maize', 0)
        infestation_rate = (infestation_count / total_classified * 100) if total_classified > 0 else 0
                
        # Calculate recent trend (compare last 7 days to previous 7 days)
        now = datetime.now()
        last_week_start = now - timedelta(days=7)
        previous_week_start = last_week_start - timedelta(days=7)
                
        # Query for last week
        c.execute(
            '''SELECT COUNT(*) FROM detections
                WHERE timestamp >= ? AND timestamp < ?
               AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?''',
            [last_week_start.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d'),
             UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX]
        )
        last_week_count = c.fetchone()[0]
                
        # Query for previous week
        c.execute(
            '''SELECT COUNT(*) FROM detections
                WHERE timestamp >= ? AND timestamp < ?
               AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?''',
            [previous_week_start.strftime('%Y-%m-%d'), last_week_start.strftime('%Y-%m-%d'),
             UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX]
        )
        previous_week_count = c.fetchone()[0]
                
        # Calculate percentage change
        if previous_week_count > 0:
            recent_trend = ((last_week_count - previous_week_count) / previous_week_count) * 100
        else:
            recent_trend = 0 if last_week_count == 0 else 100
                
        # Prepare time series data (daily counts for each class)
        time_series = {
            'labels': [],
            'data': {}
        }
                
        # Generate date range for the selected period
        start_date = now - timedelta(days=days)
        date_range = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days + 1)]
        time_series['labels'] = date_range
                
        # Query for daily counts by class
        for detection_class in ['fall-armyworm-larval-damage', 'fall-armyworm-egg', 'fall-armyworm-frass', 'healthy-maize', 'unknown']:
            daily_counts = []
                        
            for date in date_range:
                query = '''SELECT COUNT(*) FROM detections
                            WHERE date(timestamp) = ?
                            AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?'''
                params = [date, UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX]
                                
                if detection_class != 'unknown':
                    query += ' AND class = ?'
                    params.append(detection_class)
                else:
                    query += ' AND (class IS NULL OR class = "unknown")'
                                
                if class_filter:
                    query += ' AND class = ?'
                    params.append(class_filter)
                                
                if district_filter:
                    query += ' AND district = ?'
                    params.append(district_filter)
                                
                c.execute(query, params)
                count = c.fetchone()[0]
                daily_counts.append(count)
                        
            time_series['data'][detection_class] = daily_counts
                
        # Get district counts
        district_counts = {}
        c.execute(
            '''SELECT district, COUNT(*) as count
                FROM detections
                WHERE timestamp >= datetime('now', ?)
                AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?
                AND district IS NOT NULL AND district != ""
                GROUP BY district
                ORDER BY count DESC''',
            [f'-{days} days', UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX]
        )
        for row in c.fetchall():
            district_counts[row['district']] = row['count']
                
        # Get district-class breakdown
        district_class_data = {}
                
        # First, get all districts (including those with no detections)
        all_districts = list(UGANDA_DISTRICT_COORDS.keys())
        
        # For each district, get the breakdown by class
        for district in all_districts:
            district_class_data[district] = {}
                        
            for detection_class in ['fall-armyworm-larval-damage', 'fall-armyworm-egg', 'fall-armyworm-frass', 'healthy-maize', 'unknown']:
                query = '''SELECT COUNT(*) FROM detections
                            WHERE district = ? AND timestamp >= datetime('now', ?)
                            AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?'''
                params = [district, f'-{days} days', UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX]
                                
                if detection_class != 'unknown':
                    query += ' AND class = ?'
                    params.append(detection_class)
                else:
                    query += ' AND (class IS NULL OR class = "unknown")'
                                
                c.execute(query, params)
                count = c.fetchone()[0]
                                
                if count > 0:  # Only include non-zero counts
                    district_class_data[district][detection_class] = count
                
        # Close database connection
        conn.close()
                
        # Prepare response data
        response_data = {
            'total_detections': total_detections,
            'districts_affected': districts_affected,
            'infestation_rate': round(infestation_rate, 1),
            'recent_trend': round(recent_trend, 1),
            'class_distribution': class_distribution,
            'time_series': time_series,
            'district_counts': district_counts,
            'district_class_data': district_class_data
        }
                
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/uganda_districts', methods=['GET'])
def api_uganda_districts():
    """API endpoint to get all Uganda districts for the Flutter app"""
    try:
        # First get districts from the database
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute('SELECT DISTINCT district FROM detections WHERE district IS NOT NULL AND district != "" ORDER BY district')
        db_districts = [row[0] for row in c.fetchall()]
        conn.close()
        
        # Combine with our predefined districts
        all_districts = sorted(set(db_districts + list(UGANDA_DISTRICT_COORDS.keys())))
        
        return jsonify(all_districts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
                   AND district != "" 
                   ORDER BY district'''
                
        c.execute(query, [UGANDA_LAT_MIN, UGANDA_LAT_MAX, UGANDA_LON_MIN, UGANDA_LON_MAX])
        db_districts = [district[0] for district in c.fetchall()]
        conn.close()
                
        # Combine with our predefined districts
        all_districts = sorted(set(db_districts + list(UGANDA_DISTRICT_COORDS.keys())))
        
        return jsonify(all_districts)
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
