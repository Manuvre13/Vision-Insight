// constants.js
export const API_CONFIG = {
    // Change this to your computer's local IP address when testing
    // Example: 'http://192.168.1.100:8000'
    BASE_URL: 'https://huggingface.co/spaces/Manuvre/vision-insight-api',  
    ENDPOINTS: {
      PREDICT: '/predict',
      HEALTH: '/health'
    },
    TIMEOUT: 30000
  };
  
  export const DISEASE_LABELS = {
    'N': 'Normal',
    'D': 'Diabetic Retinopathy',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'Age-related Macular Degeneration'
  };