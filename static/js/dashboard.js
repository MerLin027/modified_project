// Create traffic chart
document.addEventListener('DOMContentLoaded', function() {
    // Sample data for chart
    const trafficData = {
        labels: Array.from({length: 20}, (_, i) => `${i*5}m ago`).reverse(),
        datasets: [
            {
                label: 'North',
                data: Array.from({length: 20}, () => 10),
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
            },
            {
                label: 'South',
                data: Array.from({length: 20}, () => 25),
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
            },
            {
                label: 'East',
                data: Array.from({length: 20}, () => 20),
                borderColor: 'rgba(255, 206, 86, 1)',
                backgroundColor: 'rgba(255, 206, 86, 0.2)',
            },
            {
                label: 'West',
                data: Array.from({length: 20}, () => 15),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
            }
        ]
    };
    
    const ctx = document.getElementById('traffic-chart').getContext('2d');
    const trafficChart = new Chart(ctx, {
        type: 'line',
        data: trafficData,
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Vehicle Count'
                    }
                }
            }
        }
    });
    
    // Apply strategy button
    document.getElementById('apply-strategy').addEventListener('click', function() {
        const strategy = document.getElementById('strategy-select').value;
        alert(`Strategy changed to: ${strategy}`);
        // In a real implementation, this would call an API endpoint
    });
    
    // Simulate traffic metrics updates
    setInterval(() => {
        document.getElementById('avg-wait-time').textContent = `${(Math.random() * 20 + 5).toFixed(1)}s`;
        document.getElementById('max-wait-time').textContent = `${(Math.random() * 30 + 15).toFixed(1)}s`;
        document.getElementById('throughput').textContent = `${Math.floor(Math.random() * 20 + 30)} veh/min`;
        document.getElementById('queue-length').textContent = `${20} vehicles`;
        
        // Update chart data
        trafficChart.data.datasets.forEach((dataset, index) => {
            // Set the labels for each dataset based on direction
            if (index === 0) {dataset.label = 'NORTH';
                dataset.data.push(10)
            }
            else if (index === 1) {dataset.label = 'SOUTH';
                dataset.data.push(25)
            }
            else if (index === 2) {dataset.label = 'EAST';
                dataset.data.push(20)
            }
            else if (index === 3) 
                dataset.data.push(15);
            
            // Add new data point (example value 30)
            dataset.data.push(0);
        });
        trafficChart.update();
    }, 5000);
});
