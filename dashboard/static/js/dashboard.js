const ws = new WebSocket('ws://localhost:8000/ws');
const chart = initChart();
const maxDataPoints = 50;

function initChart() {
    const ctx = document.getElementById('telemetryChart').getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Voltage (V)',
                data: [],
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                yAxisID: 'y'
            }, {
                label: 'Temperature (°C)',
                data: [],
                borderColor: '#4ecdc4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                yAxisID: 'y1'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    ticks: { color: 'white' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    ticks: { color: 'white' },
                    grid: { drawOnChartArea: false, color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    ticks: { color: 'white' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: 'white' }
                }
            }
        }
    });
}

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    updateUI(data);
    updateChart(data);
    
    if (data.is_anomaly) {
        showAlert('Anomaly detected!', 'critical');
    }
};

function updateUI(data) {
    document.getElementById('voltage').textContent = data.voltage;
    document.getElementById('current').textContent = data.current;
    document.getElementById('temperature').textContent = data.temperature;
    document.getElementById('soc').textContent = data.soc;
}

function updateChart(data) {
    chart.data.labels.push(data.timestamp);
    chart.data.datasets[0].data.push(data.voltage);
    chart.data.datasets[1].data.push(data.temperature);
    
    if (chart.data.labels.length > maxDataPoints) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }
    
    chart.update('none');
}

function showAlert(message, type) {
    const alertsDiv = document.getElementById('alerts');
    const alert = document.createElement('div');
    alert.className = `alert ${type}`;
    alert.textContent = message;
    
    alertsDiv.appendChild(alert);
    alertsDiv.style.display = 'block';
    
    setTimeout(() => {
        alert.remove();
        if (alertsDiv.children.length === 0) {
            alertsDiv.style.display = 'none';
        }
    }, 5000);
}

ws.onopen = function() {
    console.log('Connected to EV telemetry stream');
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};