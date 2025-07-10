# MedBillGuard Production Frontend

A comprehensive web dashboard for testing and using the MedBillGuard medical bill analysis system deployed on Railway.

## Features

### 🌐 Environment Management
- **Environment Toggle**: Switch between local development and Railway production
- **Real-time Status**: Live monitoring of API health and availability
- **Smart URL Detection**: Automatically connects to the correct backend

### 📊 System Monitoring
- **API Health Check**: Real-time backend service status
- **Agent Status**: Monitor available analysis agents
- **System Metrics**: View analysis statistics and performance data
- **Uptime Tracking**: Server uptime and version information

### 🧪 Comprehensive Testing
- **Quick Tests**: One-click testing of all major endpoints
- **Health Check Test**: Verify API connectivity
- **Agent List Test**: Check available agents
- **Metrics Test**: Verify monitoring endpoints
- **Quick Analysis Test**: Test analysis pipeline with sample data

### 📄 Document Analysis
- **File Upload**: Support for PDF, JPEG, PNG files up to 15MB
- **Multi-language Support**: English, Hindi, Bengali, Tamil
- **State-specific Rates**: Support for different state healthcare rates
- **Insurance Types**: CGHS, ESI, Private insurance validation
- **Real-time Results**: Live analysis with detailed feedback

### 🎨 Modern UI/UX
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Live status indicators and progress bars
- **Error Handling**: Clear error messages and troubleshooting
- **Accessibility**: Screen reader compatible with ARIA labels

## Quick Start

### 1. Start the Production Dashboard

```bash
# From the VivaranAI directory
cd frontend

# Start the production server
python serve-production.py

# Or specify a custom port
python serve-production.py 8080

# Auto-open in browser
python serve-production.py --open
```

### 2. Access the Dashboard

Open your browser to: `http://localhost:3000`

### 3. Test the System

1. **Check System Status**: The dashboard automatically checks API health on load
2. **Run Quick Tests**: Click any of the quick test buttons to verify functionality
3. **Upload a Document**: Use the file upload section to analyze a real medical bill
4. **Monitor Performance**: View real-time metrics and agent status

## Production URLs

- **Railway Backend**: `https://endearing-prosperity-production.up.railway.app`
- **API Health**: `https://endearing-prosperity-production.up.railway.app/health`
- **API Docs**: `https://endearing-prosperity-production.up.railway.app/docs`

## Testing Guide

### Basic Health Check
```bash
# The dashboard will automatically test:
curl https://endearing-prosperity-production.up.railway.app/health
```

### Quick Analysis Test
The dashboard includes a built-in quick test that:
1. Creates a sample medical bill
2. Encodes it to base64
3. Sends it to the `/analyze` endpoint
4. Displays the complete response

### File Upload Testing
1. Use the sample bills in `frontend/sample_bills/`
2. Upload through the dashboard
3. Monitor the analysis process
4. View detailed results

## Environment Configuration

### Development Mode
- **Backend**: `http://localhost:8001`
- **Purpose**: Testing with local development server
- **Features**: Full debugging, hot reload

### Production Mode (Default)
- **Backend**: `https://endearing-prosperity-production.up.railway.app`
- **Purpose**: Testing with Railway deployment
- **Features**: Real production environment, live data

## API Endpoints Tested

| Endpoint | Purpose | Test Available |
|----------|---------|----------------|
| `/health` | System health check | ✅ |
| `/agents` | List available agents | ✅ |
| `/metrics` | System metrics | ✅ |
| `/metrics/summary` | Performance summary | ✅ |
| `/analyze` | Document analysis | ✅ |
| `/analyze-enhanced` | Enhanced analysis | ✅ |

## Error Handling

The dashboard includes comprehensive error handling:
- **Network Issues**: Automatic retry and fallback
- **API Errors**: Detailed error messages with troubleshooting tips
- **File Validation**: Size and format checking
- **Response Validation**: Data integrity checks

## Performance Features

- **Async Operations**: Non-blocking UI updates
- **Loading States**: Clear progress indicators
- **Response Time Tracking**: Performance monitoring
- **Caching**: Optimized for repeated requests

## Browser Compatibility

- **Chrome**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support
- **Mobile**: Responsive design

## Security Features

- **CORS Headers**: Proper cross-origin handling
- **Input Validation**: File size and type checking
- **Error Sanitization**: Safe error message display
- **No Credential Storage**: Secure by design

## Troubleshooting

### Common Issues

1. **"API Offline" Status**
   - Check internet connection
   - Verify Railway deployment is running
   - Try switching to Development mode for local testing

2. **File Upload Fails**
   - Ensure file is under 15MB
   - Check file format (PDF, JPEG, PNG only)
   - Verify base64 encoding is working

3. **Analysis Takes Too Long**
   - Railway cold starts can take 30-60 seconds
   - Check system metrics for performance issues
   - Try the quick test first

### Debug Mode

Enable browser developer tools to see:
- Network requests and responses
- Console logs for debugging
- Performance metrics

## Development

### Local Development
```bash
# Start local backend first
python agents/server.py

# Then start frontend in development mode
cd frontend
python serve-production.py
# Switch to Development mode in the UI
```

### Customization

The dashboard is built with:
- **Alpine.js**: For reactive components
- **Tailwind CSS**: For styling
- **Font Awesome**: For icons
- **Modern JavaScript**: ES6+ features

## Support

For issues or questions:
1. Check the Railway logs: `railway logs`
2. Test individual endpoints manually
3. Verify the backend is running
4. Check network connectivity

## Features in Development

- [ ] Real-time WebSocket updates
- [ ] Batch file processing
- [ ] Advanced analytics dashboard
- [ ] User authentication
- [ ] Bill comparison features
- [ ] Export/import functionality

---

**Note**: This frontend is designed specifically for testing the Railway deployment. For production use, additional security measures and authentication would be required. 