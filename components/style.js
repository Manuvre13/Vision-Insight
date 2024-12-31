import { StyleSheet, Platform } from 'react-native';

export const lightColors = {
  primary: '#4a90e2',
  secondary: '#f5f5f5',
  text: '#333333',
  error: '#ff6b6b',
  success: '#51cf66',
  warning: '#ffd43b',
  progressBar: '#4CAF50',
  progressBackground: '#e0e0e0',
  background: '#ffffff',
  cardBackground: '#f5f5f5',
};

export const darkColors = {
  primary: '#60a5fa',
  secondary: '#374151',
  text: '#e5e7eb',
  error: '#ef4444',
  success: '#10b981',
  warning: '#f59e0b',
  progressBar: '#34d399',
  progressBackground: '#4b5563',
  background: '#1f2937',
  cardBackground: '#374151',
};

export const createStyles = (colors) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  contentContainer: {
    padding: 20,
    width: '100%',
    ...(Platform.OS === 'web' ? {
      maxWidth: 800,
      alignSelf: 'center',
    } : {}),
  },
  button: {
    backgroundColor: colors.primary,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    marginVertical: 10,
    width: '100%',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  imagePreview: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 8,
    marginVertical: 20,
    ...(Platform.OS === 'web' ? {
      maxWidth: 500,
      alignSelf: 'center',
    } : {}),
  },
  loadingContainer: {
    padding: 20,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: colors.text,
    fontSize: 16,
  },
  predictionContainer: {
    width: '100%',
    backgroundColor: colors.cardBackground,
    borderRadius: 8,
    padding: 15,
    marginTop: 20,
  },
  predictionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    color: colors.text,
  },
  metricsContainer: {
    marginBottom: 15,
    padding: 10,
    backgroundColor: colors.background,
    borderRadius: 8,
  },
  metricText: {
    fontSize: 14,
    color: colors.text,
    marginBottom: 5,
  },
  predictionItem: {
    marginBottom: 15,
  },
  predictionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 5,
  },
  predictionLabel: {
    fontSize: 16,
    color: colors.text,
    marginBottom: 5,
  },
  predictionValue: {
    fontSize: 14,
    color: colors.text,
    fontWeight: 'bold',
  },
  progressBarContainer: {
    height: 20,
    backgroundColor: colors.progressBackground,
    borderRadius: 10,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: colors.progressBar,
  },
  serverWarning: {
    backgroundColor: colors.warning,
    padding: 10,
    borderRadius: 8,
    marginBottom: 15,
  },
  warningText: {
    color: colors.text,
    textAlign: 'center',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    color: colors.error,
    fontSize: 16,
    textAlign: 'center',
  },
  themeToggle: {
    ...Platform.select({
      web: {
        position: 'relative', // Change from 'absolute' to 'relative'
        alignSelf: 'flex-end', // Align to the right
        marginBottom: 15, // Add some margin to separate from other elements
        padding: 10,
        borderRadius: 8,
        backgroundColor: colors.primary,
        zIndex: 1000,
      },
      default: {
        alignSelf: 'flex-end',
        marginBottom: 10,
        marginHorizontal: 20,
        padding: 10,
        borderRadius: 8,
        backgroundColor: colors.primary,
        zIndex: 1000,
      }
    }),
  },
});
