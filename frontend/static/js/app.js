// API Base URL
const API_BASE = window.location.origin;

// Tab Navigation
function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // Remove active class from all buttons
    const buttons = document.querySelectorAll('.nav-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    // Load tab-specific data
    if (tabName === 'analytics') {
        loadAnalytics();
    }
}

// Load System Status
async function loadStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        document.getElementById('status-text').textContent = 
            `System ${data.status} - Version ${data.version}`;
    } catch (error) {
        console.error('Error loading status:', error);
        document.getElementById('status-text').textContent = 'Error loading status';
    }
}

// Load Data Summary
async function loadDataSummary() {
    try {
        const response = await fetch(`${API_BASE}/api/data/summary`);
        const data = await response.json();
        
        // Update quick stats
        document.getElementById('data-points').textContent = data.shape[0];
        document.getElementById('features').textContent = data.shape[1];
        
        // Update summary data
        document.getElementById('summary-data').textContent = 
            JSON.stringify(data.summary, null, 2);
    } catch (error) {
        console.error('Error loading data summary:', error);
        document.getElementById('summary-data').textContent = 'Error loading data';
    }
}

// Load Analytics
async function loadAnalytics() {
    try {
        // Load visualization data
        const response = await fetch(`${API_BASE}/api/visualizations/sample`);
        const data = await response.json();
        
        // Create Chart.js visualization
        const ctx = document.getElementById('mainChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.data.dates,
                datasets: [
                    {
                        label: 'Category A',
                        data: data.data.category_a,
                        borderColor: 'rgb(102, 126, 234)',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Category B',
                        data: data.data.category_b,
                        borderColor: 'rgb(118, 75, 162)',
                        backgroundColor: 'rgba(118, 75, 162, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Sample Data Visualization'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        // Create Plotly time series
        const trace = {
            x: data.data.dates,
            y: data.data.values,
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: 'rgb(102, 126, 234)' },
            line: { color: 'rgb(102, 126, 234)' }
        };
        
        const layout = {
            title: 'Time Series Analysis',
            xaxis: { title: 'Date' },
            yaxis: { title: 'Value' },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)'
        };
        
        Plotly.newPlot('timeseries-plot', [trace], layout, {responsive: true});
        
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// Handle Prediction Form
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const features = [
                parseFloat(document.getElementById('feature1').value),
                parseFloat(document.getElementById('feature2').value),
                parseFloat(document.getElementById('feature3').value)
            ];
            
            try {
                const response = await fetch(`${API_BASE}/api/predict`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ features: features })
                });
                
                const data = await response.json();
                
                // Show result
                const resultBox = document.getElementById('prediction-result');
                const output = document.getElementById('prediction-output');
                
                output.innerHTML = `
                    <p><strong>Class:</strong> ${data.prediction.class}</p>
                    <p><strong>Confidence:</strong> ${(data.prediction.confidence * 100).toFixed(2)}%</p>
                    <p><strong>Timestamp:</strong> ${data.timestamp}</p>
                `;
                
                resultBox.style.display = 'block';
            } catch (error) {
                console.error('Error making prediction:', error);
                alert('Error making prediction. Please try again.');
            }
        });
    }
    
    // Load initial data
    loadStatus();
    loadDataSummary();
});

// Test API Endpoint
async function testAPI() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        const resultBox = document.getElementById('api-result');
        const output = document.getElementById('api-output');
        
        output.textContent = JSON.stringify(data, null, 2);
        resultBox.style.display = 'block';
    } catch (error) {
        console.error('Error testing API:', error);
        alert('Error testing API. Please try again.');
    }
}
