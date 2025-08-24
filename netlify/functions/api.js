// Netlify Function to proxy API requests to your local backend
// This allows you to update the backend URL in one place

const BACKEND_URL = process.env.BACKEND_URL || 'https://f3b96af8b7bb.ngrok-free.app';

exports.handler = async (event, context) => {
  // Allow CORS
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, ngrok-skip-browser-warning',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  };

  // Handle preflight requests
  if (event.httpMethod === 'OPTIONS') {
    return {
      statusCode: 200,
      headers,
      body: '',
    };
  }

  try {
    // Extract the API path from the request
    const apiPath = event.path.replace('/.netlify/functions/api', '');
    const targetUrl = `${BACKEND_URL}${apiPath}`;

    // Prepare the request
    const requestOptions = {
      method: event.httpMethod,
      headers: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true',
        ...event.headers,
      },
    };

    // Add body for non-GET requests
    if (event.body && event.httpMethod !== 'GET') {
      requestOptions.body = event.body;
    }

    // Add query parameters
    const url = new URL(targetUrl);
    if (event.queryStringParameters) {
      Object.keys(event.queryStringParameters).forEach(key => {
        url.searchParams.append(key, event.queryStringParameters[key]);
      });
    }

    // Make the request to your backend
    const response = await fetch(url.toString(), requestOptions);
    const data = await response.text();

    return {
      statusCode: response.status,
      headers: {
        ...headers,
        'Content-Type': response.headers.get('content-type') || 'application/json',
      },
      body: data,
    };

  } catch (error) {
    console.error('Proxy error:', error);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: 'Backend proxy error', message: error.message }),
    };
  }
};
