function showRegister() {
    document.getElementById('login-container').style.display = 'none';
    document.getElementById('register-container').style.display = 'flex';
}

function showLogin() {
    document.getElementById('register-container').style.display = 'none';
    document.getElementById('login-container').style.display = 'flex';
}

// Function to show notification
function showNotification(message) {
    const notification = document.getElementById('notification');
    const notificationMessage = document.getElementById('notification-message');
    notificationMessage.textContent = message;
    notification.style.display = 'flex';
    // Hide after 3 seconds
    setTimeout(() => {
        notification.style.display = 'none';
    }, 3000);
}

async function handleRegister() {
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;

    const response = await fetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password })
    });
    const result = await response.json();
    
    if (response.ok) {
        showNotification('Welcome! Your registration was successful. Let\'s get started.');
        showLogin();
    } else {
        showNotification(result.message); // Show error message (e.g., "User already exists")
    }
}

async function handleLogin() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
    const result = await response.json();
    
    if (response.ok) {
        showNotification('Welcome! You have successfully logged in.');
        document.getElementById('signup-section').style.display = 'none';
        localStorage.setItem('user_id', result.user_id);
        setTimeout(() => {
            window.location.href = '/chatbot';
        }, 2000); // Redirect after showing notification
    } else {
        showNotification(result.message); // Show error message (e.g., "Incorrect password")
    }
}