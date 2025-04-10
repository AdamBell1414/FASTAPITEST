// Define Uganda's bounds
const ugandaSW = L.latLng(-1.5, 29.5);  // Southwest corner
const ugandaNE = L.latLng(4.2, 35.0);   // Northeast corner
const ugandaBounds = L.latLngBounds(ugandaSW, ugandaNE);

// Initialize the map centered on Uganda with restricted bounds
const map = L.map('map', {
    center: [1.3733, 32.2903],
    zoom: 7,
    maxBounds: ugandaBounds.pad(0.1), // Add a small padding (10%) to the bounds
    minZoom: 6,
    maxZoom: 12
});

// Restrict panning to Uganda's bounds
map.on('drag', function() {
    map.panInsideBounds(ugandaBounds, { animate: false });
});

// Add tile layer (using OpenStreetMap)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    bounds: ugandaBounds
}).addTo(map);

// Add a border to highlight Uganda's approximate borders
const ugandaOutline = L.rectangle(ugandaBounds, {
    color: "#3388ff",
    weight: 2,
    fill: false,
    fillOpacity: 0.1
}).addTo(map);

// Add a title to the map
const title = L.control({position: 'topleft'});
title.onAdd = function(map) {
    const div = L.DomUtil.create('div', 'info title');
    div.innerHTML = '<h3>Fall Armyworm Detections in Uganda</h3>';
    return div;
};
title.addTo(map);

// Legend control
const legend = L.control({position: 'bottomright'});
legend.onAdd = function(map) {
    const div = L.DomUtil.create('div', 'info legend');
    const classes = {
        'fall-armyworm-larval-damage': 'red',
        'fall-armyworm-egg': 'orange',
        'fall-armyworm-frass': 'yellow',
        'healthy-maize': 'green',
        'unknown': 'gray'
    };
    
    let html = '<div><strong>Legend</strong></div>';
    for (const key in classes) {
        html +=
            '<i style="background:' + classes[key] + '"></i> ' +
            key.replace(/-/g, ' ') + '<br>';
    }
    div.innerHTML = html;
    return div;
};
legend.addTo(map);

// Add a district summary panel
const districtSummary = L.control({position: 'topright'});
districtSummary.onAdd = function(map) {
    const div = L.DomUtil.create('div', 'info district-summary');
    div.innerHTML = '<h4>Uganda Districts Summary</h4><div id="district-list">Loading data...</div>';
    return div;
};
districtSummary.addTo(map);

// Add a filter control
const filterControl = L.control({position: 'topright'});
filterControl.onAdd = function(map) {
    const div = L.DomUtil.create('div', 'info filter-control');
    div.innerHTML = `
        <h4>Filter Data</h4>
        <div class="filter-options">
            <label>
                <select id="district-filter">
                    <option value="">All Districts</option>
                </select>
            </label>
            <button id="apply-filter">Apply</button>
        </div>
    `;
    return div;
};
filterControl.addTo(map);

// Function to get color based on detection class
function getColor(classType) {
    return {
        'fall-armyworm-larval-damage': 'red',
        'fall-armyworm-egg': 'orange',
        'fall-armyworm-frass': 'yellow',
        'healthy-maize': 'green'
    }[classType] || 'gray';
}

// Store markers and data globally for filtering
let allMarkers = [];
let ugandaData = [];

// Fetch data from your API
async function loadMapData(districtFilter = '') {
    try {
        // Clear existing markers
        allMarkers.forEach(marker => map.removeLayer(marker));
        allMarkers = [];
        
        // Only fetch new data if we don't have it yet
        if (ugandaData.length === 0) {
            const response = await fetch('/map_data');
            const data = await response.json();
            
            // Filter data to only include points within Uganda's bounding box
            ugandaData = data.filter(d => 
                ugandaBounds.contains(L.latLng(d.latitude, d.longitude))
            );
            
            // Populate district filter dropdown
            const districts = [...new Set(ugandaData
                .filter(d => d.district)
                .map(d => d.district))].sort();
                
            const districtSelect = document.getElementById('district-filter');
            districtSelect.innerHTML = '<option value="">All Districts</option>';
            districts.forEach(district => {
                const option = document.createElement('option');
                option.value = district;
                option.textContent = district;
                districtSelect.appendChild(option);
            });
        }
        
        // Apply district filter if specified
        const filteredData = districtFilter 
            ? ugandaData.filter(d => d.district === districtFilter)
            : ugandaData;
        
        // Create a map to count detections by district
        const districtCounts = {};
        
        filteredData.forEach(detection => {
            // Count detections by district
            if (detection.district) {
                districtCounts[detection.district] = (districtCounts[detection.district] || 0) + 1;
            }
            
            const marker = L.circleMarker(
                [detection.latitude, detection.longitude],
                {
                    radius: 8,
                    fillColor: getColor(detection.class),
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }
            ).addTo(map);
            
            marker.bindPopup(`
                <b>${detection.result}</b><br>
                Confidence: ${detection.confidence}%<br>
                District: ${detection.district || 'Unknown'}<br>
                Date: ${new Date(detection.timestamp).toLocaleString()}
            `);
            
            allMarkers.push(marker);
        });
        
        // Update district summary panel
        let districtHtml = '';
        if (Object.keys(districtCounts).length > 0) {
            // Sort districts by count (highest first)
            const sortedDistricts = Object.entries(districtCounts)
                .sort((a, b) => b[1] - a[1]);
            
            districtHtml = '<table>';
            districtHtml += '<tr><th>District</th><th>Detections</th></tr>';
            sortedDistricts.forEach(([district, count]) => {
                districtHtml += `<tr><td>${district}</td><td>${count}</td></tr>`;
            });
            districtHtml += '</table>';
        } else {
            districtHtml = '<p>No detections found</p>';
        }
        document.getElementById('district-list').innerHTML = districtHtml;
        
        // If we have data points, adjust the view but stay within Uganda
        if (filteredData.length > 0) {
            const markers = filteredData.map(d => L.marker([d.latitude, d.longitude]));
            const markerGroup = new L.featureGroup(markers);
            
            // Get bounds of markers but ensure they're within Uganda
            const dataBounds = markerGroup.getBounds();
            const visibleBounds = dataBounds.intersects(ugandaBounds) ? 
                dataBounds.intersection(ugandaBounds.pad(0.05)) : ugandaBounds;
                
            map.fitBounds(visibleBounds, {
                padding: [50, 50],
                maxZoom: 10
            });
        } else if (districtFilter) {
            // If filtering by district but no points, keep the current view
        } else {
            // Reset to full Uganda view if no data at all
            resetView();
        }
    } catch (error) {
        console.error('Error loading map data:', error);
        document.getElementById('district-list').innerHTML = '<p>Error loading data</p>';
    }
}

// Add a reset view button
const resetButton = L.control({position: 'bottomleft'});
resetButton.onAdd = function(map) {
    const div = L.DomUtil.create('div', 'info reset-button');
    div.innerHTML = '<button onclick="resetView()">Reset View</button>';
    return div;
};
resetButton.addTo(map);

// Function to reset the view to Uganda
function resetView() {
    map.fitBounds(ugandaBounds);
    document.getElementById('district-filter').value = '';
    loadMapData();
}

// Set up event listeners after DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Load initial data
    loadMapData();
    
    // Set up filter button
    document.getElementById('apply-filter').addEventListener('click', () => {
        const districtFilter = document.getElementById('district-filter').value;
        loadMapData(districtFilter);
    });
});
