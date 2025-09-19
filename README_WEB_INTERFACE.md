# Arogya AI - Web Interface

## 🌐 Web Application Overview

The Arogya AI web interface provides a user-friendly way to access disease prediction with Ayurvedic recommendations through a modern, responsive web application.

## ✨ Features

- **Intuitive Form Interface**: Easy-to-use form similar to Symptoma AI
- **Comprehensive Input Fields**: All necessary health and lifestyle information
- **Real-time Predictions**: Instant disease prediction with 99.5% accuracy
- **Beautiful Results Display**: Well-organized Ayurvedic recommendations
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Print Functionality**: Print results for offline reference

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web Application
```bash
python app.py
```

### 3. Open in Browser
Navigate to: `http://localhost:5000`

## 📋 Form Fields

### Basic Information (Required)
- **Symptoms**: Describe your symptoms (comma-separated)
- **Age**: Your age in years
- **Height**: Your height in centimeters
- **Weight**: Your weight in kilograms
- **Gender**: Male or Female

### Ayurvedic Constitution & Lifestyle (Optional)
- **Body Type (Dosha)**: Your Ayurvedic constitution
  - Vata (Air & Space - Thin, energetic, creative)
  - Pitta (Fire & Water - Medium build, focused, warm)
  - Kapha (Earth & Water - Sturdy, calm, stable)
  - Mixed constitutions available
- **Food Habits**: Vegetarian, Non-Vegetarian, Vegan, Mixed
- **Current Medications**: List any current medications
- **Known Allergies**: List any known allergies
- **Current Season**: Spring, Summer, Monsoon, Autumn, Winter
- **Current Weather**: Hot, Cold, Humid, Dry, Rainy

## 📊 Results Display

Each prediction provides comprehensive information:

### Medical Prediction
- Disease name with confidence percentage
- User's body type (constitution)

### Ayurvedic Treatment Plan
- **Recommended Herbs**: Sanskrit and English names with therapeutic effects
- **Recommended Therapies**: Sanskrit and English therapy names with benefits
- **Dietary Recommendations**: Personalized dietary guidance
- **Body Type Effects**: How treatments specifically benefit your constitution

## 🛠️ Technical Implementation

### Backend (Flask)
- **Framework**: Flask 3.1.2
- **API Endpoints**:
  - `GET /`: Main web interface
  - `POST /predict`: Prediction API
  - `GET /health`: Health check
- **AI Integration**: Uses existing ArogyaAI prediction system

### Frontend
- **HTML5**: Semantic, accessible markup
- **CSS3**: Modern styling with gradients and animations
- **JavaScript**: Interactive form handling and AJAX requests
- **Bootstrap 5**: Responsive design framework
- **Font Awesome**: Icons for better UX

### File Structure
```
├── app.py              # Flask web application
├── templates/
│   └── index.html      # Main web interface
├── static/
│   ├── css/
│   │   └── style.css   # Custom styles
│   └── js/
│       └── script.js   # Frontend JavaScript
└── arogya_predict.py   # AI prediction system
```

## 🔗 API Usage

### Prediction API
```bash
POST /predict
Content-Type: application/json

{
  "symptoms": "fever, headache, body ache",
  "age": "30",
  "height": "170",
  "weight": "70",
  "gender": "Female",
  "dosha": "Pitta",
  "food_habits": "Vegetarian",
  "current_medication": "None",
  "allergies": "None",
  "season": "Summer",
  "weather": "Hot"
}
```

### Response Format
```json
{
  "success": true,
  "prediction": {
    "disease": "Jwara (Fever)",
    "confidence": "100.0%",
    "user_body_type": "Pitta"
  },
  "ayurvedic_recommendations": {
    "herbs_sanskrit": "Tulasi, Sunthi, Marich...",
    "herbs_english": "Holy Basil, Dry Ginger...",
    "herbs_effects": "Antipyretic, immune-boosting...",
    "therapies_sanskrit": "Langhana, Swedana...",
    "therapies_english": "Fasting therapy, Steam therapy...",
    "therapies_effects": "Reduces body heat, promotes sweating...",
    "dietary_recommendations": "Light, easily digestible foods...",
    "body_type_effects": "Balances aggravated Pitta dosha..."
  }
}
```

## 🎨 User Experience Features

- **Loading Animation**: Visual feedback during prediction
- **Form Validation**: Real-time input validation
- **Error Handling**: User-friendly error messages
- **Responsive Layout**: Optimized for all screen sizes
- **Print Support**: Print-friendly results layout
- **Accessibility**: ARIA labels and semantic markup

## 🔧 Customization

### Styling
Modify `static/css/style.css` to customize:
- Colors and themes
- Layout and spacing
- Animations and transitions

### Form Fields
Update `templates/index.html` to:
- Add new form fields
- Modify dropdown options
- Change form layout

### Backend Logic
Extend `app.py` to:
- Add new API endpoints
- Implement additional features
- Integrate with databases

## 📱 Mobile Responsiveness

The interface is fully responsive and optimized for:
- **Desktop**: Full-width layout with side-by-side fields
- **Tablet**: Responsive grid with optimized spacing
- **Mobile**: Single-column layout with touch-friendly inputs

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
Use a production WSGI server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🛡️ Security Notes

- Input validation on both frontend and backend
- Error handling prevents system information leakage
- CSRF protection recommended for production
- Rate limiting suggested for API endpoints

## 📈 Performance

- **Fast Loading**: Optimized CSS and JavaScript
- **Efficient API**: Minimal data transfer
- **Cached Resources**: Static file caching
- **Responsive Design**: Mobile-optimized performance

## 🔍 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📝 License

This web interface is part of the Arogya AI project and follows the same licensing terms.

## 🆘 Support

For technical support or questions about the web interface:
1. Check the console for JavaScript errors
2. Verify the Flask backend is running
3. Ensure all dependencies are installed
4. Check network connectivity for API calls

---

**Experience the power of Ayurvedic AI through our intuitive web interface! 🌿**