document.addEventListener('DOMContentLoaded', function() {
    // Fetch anomalies.json and display anomaly count on the dashboard
    fetch('/home/honour/Documents/DeepAI/data_generator/data/anomalies.json')
      .then(response => response.json())
      .then(data => {
        const anomalyCount = data.length;  // Get the count of anomalies
        // Set the anomaly count in the dashboard
        document.getElementById('anomaly-counter').textContent = anomalyCount;
  
        // Redirect to the anomaly details page on button click
        document.getElementById('more-info-btn').addEventListener('click', function() {
          localStorage.setItem('anomalies', JSON.stringify(data)); // Store anomalies in local storage
          window.location.href = '/home/honour/Documents/DeepAI/web/details.html';  // Redirect to the details page
        });
      })
      .catch(error => {
        console.error('Error fetching anomalies:', error);
      });
  });
  