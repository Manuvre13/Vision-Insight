import React, { 
  useState, 
  useEffect, 
  useCallback, 
  memo,
  createContext,
  useContext 
} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Image,
  Alert,
  Platform,
  Dimensions
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

// Import Styles and Config
import { createStyles, lightColors, darkColors } from './style';
import { API_CONFIG, DISEASE_LABELS } from '../config/constants';

// Create Theme Context
const ThemeContext = createContext();

// Theme Provider Component
export const ThemeProvider = ({ children }) => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const toggleTheme = () => setIsDarkMode(!isDarkMode);
  
  const theme = {
    isDarkMode,
    toggleTheme,
    colors: isDarkMode ? darkColors : lightColors,
    styles: createStyles(isDarkMode ? darkColors : lightColors),
  };

  return (
    <ThemeContext.Provider value={theme}>
      {children}
    </ThemeContext.Provider>
  );
};

// Hook to use theme
const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// Prediction Item Component with percentage display
const PredictionItem = memo(({ diseaseName, value }) => {
  const { styles } = useTheme();
  return (
    <View style={styles.predictionItem}>
      <View style={styles.predictionHeader}>
        <Text style={styles.predictionLabel}>{diseaseName}</Text>
        <Text style={styles.predictionValue}>{(value * 100).toFixed(2)}%</Text>
      </View>
      <View style={styles.progressBarContainer}>
        <View 
          style={[
            styles.progressBar, 
            { width: Platform.select({ web: `${value * 100}%`, default: `${value * 100}%` }) }
          ]} 
        />
      </View>
    </View>
  );
});

// Error Boundary Component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error Boundary caught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      const { styles } = this.context;
      return (
        <View style={styles.errorContainer}>
          <Text style={styles.errorText}>
            An unexpected error occurred. Please try again.
          </Text>
        </View>
      );
    }
    return this.props.children;
  }
}
ErrorBoundary.contextType = ThemeContext;

// API Client
const api = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT,
  headers: {
    'Accept': 'application/json',
  }
});

// Main Component
const EyeConditionDetector = () => {
  const { colors, styles, isDarkMode, toggleTheme } = useTheme();
  
  // State Management
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [serverStatus, setServerStatus] = useState('checking');
  const [analysisMetrics, setAnalysisMetrics] = useState(null);
  const [windowDimensions, setWindowDimensions] = useState(Dimensions.get('window'));

  // Handle window resize for web
  useEffect(() => {
    if (Platform.OS === 'web') {
      const handleResize = () => {
        setWindowDimensions(Dimensions.get('window'));
      };

      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, []);

  // Calculate dynamic styles
  const dynamicStyles = {
    imagePreview: {
      width: Platform.OS === 'web' 
        ? Math.min(windowDimensions.width * 0.8, 500) 
        : '100%',
      height: Platform.OS === 'web' 
        ? Math.min(windowDimensions.width * 0.8, 500) 
        : undefined,
      aspectRatio: 1,
    }
  };

  // Image Picker Handler
  const pickImage = useCallback(async () => {
    try {
      if (Platform.OS === 'web') {
        // Web file picker
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = async (e) => {
          const file = e.target.files[0];
          if (file) {
            setImage(URL.createObjectURL(file));
            setPredictions(null);
          }
        };
        input.click();
      } else {
        // Native picker
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        
        if (status !== 'granted') {
          Alert.alert(
            'Permission Required', 
            'Please grant camera roll access to continue.',
            [{ text: 'OK' }]
          );
          return;
        }

        const result = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: true,
          aspect: [1, 1],
          quality: 1,
        });

        if (!result.canceled && result.assets[0].uri) {
          setImage(result.assets[0].uri);
          setPredictions(null);
        }
      }
    } catch (err) {
      console.error('Image picker error:', err);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  }, []);

  // Image Analysis Handler
  const analyzeImage = useCallback(async () => {
    if (!image) return;

    setLoading(true);
    try {
      const formData = new FormData();
      
      if (Platform.OS === 'web') {
        // Handle web file
        const response = await fetch(image);
        const blob = await response.blob();
        formData.append('file', blob, 'image.jpg');
      } else {
        // Handle native file
        const imageUri = image;
        const filename = imageUri.split('/').pop();
        
        formData.append('file', {
          uri: imageUri,
          type: 'image/jpeg',
          name: filename || 'image.jpg',
        });
      }

      const response = await api.post(
        API_CONFIG.ENDPOINTS.PREDICT, 
        formData, 
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          transformRequest: (data, headers) => {
            return data;
          },
        }
      );

      setPredictions(response.data.predictions);
      setAnalysisMetrics({
        confidence: response.data.confidence,
        processingTime: response.data.processing_time
      });
    } catch (err) {
      console.error('Analysis error:', err.response?.data || err.message);
      Alert.alert(
        'Analysis Failed', 
        err.response?.data?.detail || 'Unable to analyze image. Please try again.',
        [{ text: 'OK' }]
      );
    } finally {
      setLoading(false);
    }
  }, [image]);

  // Server Health Check
  useEffect(() => {
    const checkServerHealth = async () => {
      try {
        const response = await api.get(API_CONFIG.ENDPOINTS.HEALTH);
        setServerStatus(response.data.model_loaded ? 'ready' : 'error');
      } catch (err) {
        console.error('Health check error:', err);
        setServerStatus('error');
      }
    };

    const healthCheckInterval = setInterval(checkServerHealth, 30000);
    checkServerHealth();

    return () => clearInterval(healthCheckInterval);
  }, []);

  // Render Predictions
  const renderPredictions = useCallback(() => {
    if (!predictions) return null;

    return (
      <View style={styles.predictionContainer}>
        <Text style={styles.predictionTitle}>Analysis Results</Text>
        {analysisMetrics && (
          <View style={styles.metricsContainer}>
            <Text style={styles.metricText}>
              Confidence: {(analysisMetrics.confidence * 100).toFixed(2)}%
            </Text>
            <Text style={styles.metricText}>
              Processing Time: {analysisMetrics.processingTime.toFixed(2)}s
            </Text>
          </View>
        )}
        {Object.entries(predictions).map(([key, value]) => (
          <PredictionItem
            key={key}
            diseaseName={DISEASE_LABELS[key] || key}
            value={value}
          />
        ))}
      </View>
    );
  }, [predictions, analysisMetrics, styles]);

  // Server Status Warning
  const renderServerStatus = useCallback(() => {
    if (serverStatus === 'error') {
      return (
        <View style={styles.serverWarning}>
          <Text style={styles.warningText}>
            Server connection issues. Some features may be unavailable.
          </Text>
        </View>
      );
    }
    return null;
  }, [serverStatus, styles]);

  return (
    <ErrorBoundary>
      <View style={styles.container}>
        <ScrollView 
          style={styles.container}
          contentContainerStyle={[
            styles.contentContainer,
            Platform.OS === 'web' && { maxWidth: 800, alignSelf: 'center' }
          ]}
        >
          {Platform.OS === 'web' ? (
            <TouchableOpacity 
              style={styles.themeToggle} 
              onPress={toggleTheme}
            >
              <Text style={styles.themeToggleText}>
                {isDarkMode ? '☀️ Light' : '🌙 Dark'}
              </Text>
            </TouchableOpacity>
          ) : null}
  
          {Platform.OS !== 'web' && (
            <TouchableOpacity 
              style={styles.themeToggle} 
              onPress={toggleTheme}
            >
              <Text style={styles.themeToggleText}>
                {isDarkMode ? '☀️ Light' : '🌙 Dark'}
              </Text>
            </TouchableOpacity>
          )}
  
          {renderServerStatus()}
          
          <TouchableOpacity 
            style={[
              styles.button,
              serverStatus === 'error' && styles.buttonDisabled
            ]} 
            onPress={pickImage}
            disabled={serverStatus === 'error'}
          >
            <Text style={styles.buttonText}>
              Upload Image
            </Text>
          </TouchableOpacity>
  
          {image && (
            <Image 
              source={{ uri: image }} 
              style={[styles.imagePreview, dynamicStyles.imagePreview]}
              resizeMode="contain"
            />
          )}
  
          {image && !loading && (
            <TouchableOpacity 
              style={[
                styles.button,
                serverStatus === 'error' && styles.buttonDisabled
              ]} 
              onPress={analyzeImage}
              disabled={serverStatus === 'error'}
            >
              <Text style={styles.buttonText}>
                Analyze Image
              </Text>
            </TouchableOpacity>
          )}
  
          {loading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator 
                size="large" 
                color={colors.primary} 
              />
              <Text style={styles.loadingText}>
                Analyzing image...
              </Text>
            </View>
          )}
  
          {renderPredictions()}
        </ScrollView>
      </View>
    </ErrorBoundary>
  );
};

// Wrap the export with ThemeProvider
export default () => (
  <ThemeProvider>
    <EyeConditionDetector />
  </ThemeProvider>
);