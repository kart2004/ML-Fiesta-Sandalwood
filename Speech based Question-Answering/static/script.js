// scripts.js

document.getElementById('recordForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const button = this.querySelector('button');
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    
    button.disabled = true;
    button.innerHTML = 'Recording...';
    button.parentNode.insertBefore(spinner, button.nextSibling);
    
    const formData = new FormData(this);
    fetch('/record', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.transcription) {
            // Display the transcription with a fade-in effect
            const transcriptionContainer = document.getElementById('transcriptionContainer');
            document.getElementById('transcriptionText').textContent = data.transcription;
            transcriptionContainer.style.display = 'block';
            transcriptionContainer.style.opacity = 0;
            setTimeout(() => {
                transcriptionContainer.style.transition = 'opacity 1s';
                transcriptionContainer.style.opacity = 1;
            }, 100);
        } else {
            alert('Transcription failed. Please try again.');
        }
        
        // Remove the spinner and re-enable the button
        spinner.remove();
        button.disabled = false;
        button.innerHTML = 'ರೆಕಾರ್ಡ್ ಮಾಡಿ';
        
        // Run the script after displaying the transcription
        fetch('/run_script', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.status);
            if (data.status === 'success') {
                // Display the answer segment with a fade-in effect
                const answerContainer = document.getElementById('answerContainer');
                answerContainer.style.display = 'block';
                answerContainer.style.opacity = 0;
                setTimeout(() => {
                    answerContainer.style.transition = 'opacity 1s';
                    answerContainer.style.opacity = 1;
                }, 100);
                document.getElementById('answerAudio').load();
            } else {
                alert('Failed to generate answer segment.');
            }
        })
        .catch(() => {
            alert('An error occurred while running the script.');
        });
    })
    .catch(() => {
        // Handle network errors
        spinner.remove();
        button.disabled = false;
        button.innerHTML = 'ರೆಕಾರ್ಡ್ ಮಾಡಿ';
        alert('Network error. Please try again.');
    });
});