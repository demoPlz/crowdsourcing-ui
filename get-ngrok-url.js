// Script to automatically get current ngrok URL
async function getCurrentNgrokUrl() {
  try {
    // Try to get the current ngrok tunnel from the local API
    const response = await fetch('http://127.0.0.1:4040/api/tunnels');
    const data = await response.json();
    
    if (data.tunnels && data.tunnels.length > 0) {
      // Find the HTTPS tunnel
      const httpsTunnel = data.tunnels.find(tunnel => tunnel.public_url.startsWith('https://'));
      if (httpsTunnel) {
        return httpsTunnel.public_url;
      }
    }
    
    // Fallback to localhost if ngrok is not available
    return 'http://127.0.0.1:9000';
  } catch (error) {
    console.warn('Could not fetch ngrok URL, falling back to localhost:', error);
    return 'http://127.0.0.1:9000';
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = getCurrentNgrokUrl;
}

// Make it available globally
window.getCurrentNgrokUrl = getCurrentNgrokUrl;
