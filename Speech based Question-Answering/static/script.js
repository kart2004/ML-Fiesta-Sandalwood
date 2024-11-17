// scripts.js

document.querySelector('form').addEventListener('submit', function(e) {
    const button = this.querySelector('button');
    button.disabled = true;
    button.innerHTML = 'Recording...';
});