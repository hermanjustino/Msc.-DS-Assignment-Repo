// Initialize the map with actual values
var map = L.map('map').setView([43.70011, -79.4163], 13); // Example coordinates for Toronto, Canada
    
// Add a tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

// Load filtered GeoJSON data
var geojsonLayer = new L.GeoJSON.AJAX("../data/filtered_neighbourhoods.geojson", {
    onEachFeature: function (feature, layer) {
        if (feature.properties && feature.properties.AREA_NAME) {
            layer.bindPopup("Neighbourhood: " + feature.properties.AREA_NAME);
            layer.on('click', function() {
                console.log("Clicked on neighborhood:", feature.properties.AREA_NAME);
                loadChart(feature.properties.AREA_NAME);
            });
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

// Get the modal
var modal = document.getElementById("myModal");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

// When the user clicks on <span> (x), close the modal and destroy the chart
span.onclick = function() {
    modal.style.display = "none";
    if (radialChart) {
        radialChart.destroy();
        radialChart = null;
    }
}

// When the user clicks anywhere outside of the modal, close it and destroy the chart
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
        if (radialChart) {
            radialChart.destroy();
            radialChart = null;
        }
    }
}

// Load the neighborhood data from the JSON file
var neighborhoodData;
fetch('../data/neighborhood_data.json')
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        // Normalize the keys in the neighborhoodData object
        neighborhoodData = {};
        for (var key in data) {
            if (data.hasOwnProperty(key)) {
                neighborhoodData[key.trim().toLowerCase()] = data[key];
            }
        }
        console.log("Neighborhood data loaded:", neighborhoodData);
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
    });

var radialChart; // Declare the chart variable outside the function

// Function to generate gradient colors
function generateGradientColors(numColors) {
    var colors = [];
    var startColor = [255, 99, 132]; // Start color (red)
    var endColor = [54, 162, 235]; // End color (blue)

    for (var i = 0; i < numColors; i++) {
        var r = Math.floor(startColor[0] + (endColor[0] - startColor[0]) * (i / (numColors - 1)));
        var g = Math.floor(startColor[1] + (endColor[1] - startColor[1]) * (i / (numColors - 1)));
        var b = Math.floor(startColor[2] + (endColor[2] - startColor[2]) * (i / (numColors - 1)));
        colors.push(`rgb(${r}, ${g}, ${b})`);
    }

    return colors;
}

// Function to load the chart
function loadChart(neighborhood) {
    console.log("loadChart called with neighborhood:", neighborhood);
    if (!neighborhoodData) {
        console.error("Neighborhood data is not loaded yet.");
        return;
    }

    // Normalize the neighborhood name
    var normalizedNeighborhood = neighborhood.trim().toLowerCase();
    console.log("Normalized neighborhood:", normalizedNeighborhood);

    // Initialize arrays to store age groups and populations
    var ageGroups = [];
    var populations = [];

    // Iterate through each age group in the neighborhoodData object
    for (var ageGroup in neighborhoodData) {
        if (ageGroup.toLowerCase().includes("total")) {
            continue; // Skip the "total" age group
        }
        if (neighborhoodData.hasOwnProperty(ageGroup)) {
            var populationData = neighborhoodData[ageGroup];
            console.log(`Age Group: ${ageGroup}, Population Data:`, populationData); // Add this line

            // Print the keys of the population data object
            console.log("Keys in Population Data:", Object.keys(populationData));

            // Normalize the keys in the population data object
            var normalizedPopulationData = {};
            for (var key in populationData) {
                if (populationData.hasOwnProperty(key)) {
                    normalizedPopulationData[key.trim().toLowerCase()] = populationData[key];
                }
            }
            console.log("Normalized Population Data:", normalizedPopulationData);

            var population = normalizedPopulationData[normalizedNeighborhood];

            // Add debugging statement to check population data
            console.log(`Age Group: ${ageGroup}, Population: ${population}`);

            // Add the age group and population to the arrays
            ageGroups.push(ageGroup);
            populations.push(population);
        }
    }

    // Print the neighborhood and populations to the console
    console.log("Neighborhood:", neighborhood);
    console.log("Populations:", populations);

    // Generate gradient colors based on the number of age groups
    var backgroundColors = generateGradientColors(ageGroups.length);

    if (radialChart) {
        // Update the existing chart
        radialChart.data.labels = ageGroups;
        radialChart.data.datasets[0].data = populations;
        radialChart.data.datasets[0].backgroundColor = backgroundColors;
        radialChart.update();
    } else {
        // Create the radial chart
        var ctx = document.getElementById('radialChart').getContext('2d');
        radialChart = new Chart(ctx, {
            type: 'polarArea',
            data: {
                labels: ageGroups,
                datasets: [{
                    label: 'Population',
                    data: populations,
                    backgroundColor: backgroundColors
                }]
            },
            options: {
                scales: {
                    r: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Display the modal
    modal.style.display = "block";
}