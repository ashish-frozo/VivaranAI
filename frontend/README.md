# 🌐 MedBillGuard Frontend Dashboard

A beautiful, responsive web dashboard for testing the MedBillGuardAgent API.

## ✨ Features

- **🔄 Real-time API Status**: Monitor backend health with auto-refresh
- **📁 File Upload**: Drag & drop or click to upload medical bills (PDF, JPEG, PNG)
- **🌍 Multi-language Support**: English, Hindi, Bengali, Tamil OCR
- **🗺️ State Selection**: Choose from major Indian states for regional rate validation
- **⚡ Quick Test**: Test API with sample data (no file upload required)
- **📊 Visual Results**: Beautiful display of analysis results with:
  - Verdict badges (OK/WARNING/CRITICAL)
  - Summary cards (Bill amount, Overcharge, Confidence)
  - Detailed red flags with overcharge calculations
  - Actionable recommendations
  - Processing metrics

## 🚀 Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
# From project root
./start_testing.sh
```

### Option 2: Manual Setup
```bash
# Terminal 1: Start API server
python -m uvicorn medbillguardagent:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start dashboard
cd frontend
python serve.py

# Open browser
open http://localhost:3000
```

## 🎯 How to Test

### 1. **API Health Check**
- Dashboard automatically checks API status on load
- Green badge = API is healthy
- Red badge = API is offline
- Click refresh icon to recheck

### 2. **Quick Test**
- Click "Run Quick Test" button
- Uses sample medical bill data
- No file upload required
- Results appear in ~2-3 seconds

### 3. **File Upload Test**
- Click upload area or drag & drop a file
- Supported formats: PDF, JPEG, PNG (max 15MB)
- Select language and state
- Click "Analyze Medical Bill"
- Results appear with detailed breakdown

## 📋 Sample Test Data

The dashboard works with the included sample bill (`fixtures/sample_bill.txt`) which contains:
- **4 CBC tests** (3 duplicates = ₹1,500 overcharge)
- **2 General Medicine consultations** (1 duplicate = ₹2,500 overcharge)
- **Total bill**: ₹23,550
- **Expected overcharge**: ₹3,100
- **Expected verdict**: WARNING

## 🎨 UI Components

### **Status Card**
- Shows API connectivity status
- Auto-refreshes on page load
- Manual refresh button

### **Upload Section**
- Drag & drop file area
- File validation (size/format)
- Language selection (en/hi/bn/ta)
- State selection for regional rates
- Auto-generated document ID

### **Quick Test**
- One-click testing without file upload
- Uses predefined sample data
- Faster than file upload testing

### **Results Display**
- **Verdict Badge**: Color-coded severity (green/yellow/red)
- **Summary Cards**: Key metrics at a glance
- **Red Flags**: Detailed overcharge breakdown
- **Recommendations**: User-friendly guidance
- **Next Steps**: Actionable advice
- **Processing Stats**: Performance metrics

## 🔧 Technical Details

### **Frontend Stack**
- **HTML5**: Semantic markup
- **Tailwind CSS**: Utility-first styling
- **Alpine.js**: Reactive JavaScript framework
- **Font Awesome**: Icons
- **Vanilla JavaScript**: API communication

### **API Integration**
- RESTful API calls to `http://localhost:8000`
- CORS-enabled for cross-origin requests
- FormData for file uploads
- JSON responses with camelCase conversion
- Error handling with user-friendly messages

### **Responsive Design**
- Mobile-first approach
- Grid layouts for cards
- Collapsible sections
- Touch-friendly interactions

## 🐛 Troubleshooting

### **Dashboard Shows "API Offline"**
1. Ensure API server is running on port 8000
2. Check for CORS issues
3. Verify network connectivity

### **File Upload Fails**
1. Check file size (<15MB)
2. Verify file format (PDF/JPEG/PNG)
3. Ensure stable internet connection

### **No Results Displayed**
1. Check browser console for errors
2. Verify API response format
3. Test with Quick Test first

### **Styling Issues**
1. Ensure internet connection for CDN resources
2. Check browser compatibility
3. Clear browser cache

## 🔗 API Endpoints Used

- `GET /healthz` - Health check
- `GET /debug/example` - Quick test
- `POST /analyze` - File analysis
- `GET /api/docs` - Documentation

## 📱 Browser Support

- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

## 🎯 Next Steps

This dashboard provides a complete testing interface for the MedBillGuardAgent. For production use, consider:

- User authentication
- File storage management
- Result history
- Advanced filtering
- Export capabilities
- Real-time notifications

---

**Happy Testing!** 🧪✨ 