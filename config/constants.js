// constants.js
export const API_CONFIG = {
    // Change this to your computer's local IP address when testing
    // Example: 'http://192.168.1.100:8000'
    BASE_URL: 'https://manuvre-vision-insight-api.hf.space',  
    ENDPOINTS: {
      PREDICT: '/predict',
      HEALTH: '/health'
    },
    TIMEOUT: 60000,
    HEADERS: {
      'Accept': 'application/json',
      'Access-Control-Allow-Origin': '*'
    }
  };
  
  export const DISEASE_LABELS = {
    'N': 'Normal',
    'D': 'Diabetic Retinopathy',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration'
  };