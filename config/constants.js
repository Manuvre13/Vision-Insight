// constants.js
import axios from 'axios';  // Add this import at the top of the file

export const API_CONFIG = {
  BASE_URL: 'https://manuvre-vision-insight-api.hf.space',
  ENDPOINTS: {
    PREDICT: '/predict',
    HEALTH: '/health'
  },
  TIMEOUT: 60000,
  HEADERS: {
    'Accept': 'application/json',
    'Origin': window.location.origin,
  }
};

export const DISEASE_LABELS = {
  'N': 'Normal',
  'D': 'Diabetic Retinopathy',
  'G': 'Glaucoma',
  'C': 'Cataract',
  'A': 'Age-related Macular Degeneration'
};

// Modified API client setup
export const createApiClient = () => {
  const client = axios.create({
    baseURL: API_CONFIG.BASE_URL,
    timeout: API_CONFIG.TIMEOUT,
    headers: API_CONFIG.HEADERS,
    withCredentials: false
  });

  // Request interceptor
  client.interceptors.request.use(
    config => {
      // Add timestamp to prevent caching
      if (config.method === 'get') {
        config.params = {
          ...config.params,
          _t: new Date().getTime()
        };
      }
      return config;
    },
    error => {
      console.error('Request error:', error);
      return Promise.reject(error);
    }
  );

  // Response interceptor
  client.interceptors.response.use(
    response => response,
    error => {
      console.error('Response error:', error.response || error);
      
      // Customize error messages based on status codes
      if (error.response) {
        switch (error.response.status) {
          case 404:
            error.message = 'Service not found. Please check the API endpoint.';
            break;
          case 403:
            error.message = 'Access forbidden. Please check CORS settings.';
            break;
          case 500:
            error.message = 'Internal server error. Please try again later.';
            break;
          case 503:
            error.message = 'Service unavailable. The model might be loading.';
            break;
          default:
            error.message = 'An unexpected error occurred. Please try again.';
        }
      } else if (error.code === 'ECONNABORTED') {
        error.message = 'Request timed out. Please check your connection.';
      }
      
      return Promise.reject(error);
    }
  );

  return client;
};