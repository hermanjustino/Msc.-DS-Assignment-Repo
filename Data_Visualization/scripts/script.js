// Initialize the map with actual values
var map = L.map('map').setView([43.70011, -79.4163], 13); // Example coordinates for Toronto, Canada
    
// Add a tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

// Load filtered GeoJSON data
var geojsonLayer = new L.GeoJSON.AJAX("data/filtered_neighbourhoods.geojson", {
    onEachFeature: function (feature, layer) {
        if (feature.properties && feature.properties.AREA_NAME) {
            layer.bindPopup("Neighbourhood: " + feature.properties.AREA_NAME);
        }
    },
    style: function (feature) {
        return {
            color: "#ff7800",
            weight: 2,
            opacity: 1
        };
    }
}).addTo(map);