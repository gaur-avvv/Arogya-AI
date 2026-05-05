# 🌿 ArogyaAI: Hybrid Intelligence for Ayurvedic Clinical Support

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Firebase](https://img.shields.io/badge/firebase-%23039BE5.svg?style=for-the-badge&logo=firebase)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Vercel](https://img.shields.io/badge/vercel-%23000000.svg?style=for-the-badge&logo=vercel&logoColor=white)



https://github.com/user-attachments/assets/71080ee0-c7ad-4143-b0fb-4fe6cfd9c607



ArogyaAI is a modern, cloud-connected Clinical Decision Support System (CDSS) designed to bridge the gap between traditional Ayurvedic medicine and modern Artificial Intelligence. 

By utilizing a **Dual-Engine AI Architecture** (Deterministic Machine Learning + Generative AI) and strict **Role-Based Access Control (RBAC)**, ArogyaAI provides a secure, end-to-end ecosystem for both patients and medical practitioners. This project is configured as a full-stack monorepo with automated CI/CD and serverless deployments on Vercel.

---

## 🎯 The Problem & Solution
Traditional Ayurvedic diagnostics rely heavily on practitioner intuition, while modern medical AI models act as "black boxes" that ignore holistic factors like Doshas (Prakriti) and seasonality. Furthermore, exposing raw, low-confidence ML predictions directly to patients poses a severe ethical and psychological risk.

**ArogyaAI solves this by:**
1. Combining mathematical Random Forest predictions with Generative LLM contextual reasoning.
2. Utilizing Explainable AI (XAI) so doctors can see *why* the AI made its decision.
3. Implementing strict Clinical Safety Guardrails that mask low-confidence predictions to prevent patient panic.

---

## ✨ Key Features

### 🔐 Role-Based Architecture (Multi-Tenant)
* **Practitioner Portal:** Doctors have a comprehensive dashboard to run AI diagnostics, view clinic-wide metrics, and manage patient records.
* **Patient Portal:** Patients have a soothing, non-intimidating dashboard to log daily symptoms (Health Diaries) and view safe, actionable Ayurvedic protocols prescribed by their doctor.
* **Clinic ID Siloing:** Data is strictly isolated. Patients link their accounts to a specific doctor using a unique 6-character `Clinic ID`, ensuring secure, HIPAA-compliant-style data routing.

### 🧠 Dual-Engine AI System
* **Engine 1 (Deterministic):** A Python API backend running a trained Random Forest model (Scikit-Learn). It analyzes symptom strings via TF-IDF vectorization and outputs a disease probability and confidence score.
* **Engine 2 (Generative):** Google Gemini 2.5 LLM analyzes the patient's Dosha, age, gender, and the ML prediction to generate a holistic, personalized Ayurvedic protocol (Herbs & Lifestyle).

### 🛡️ Clinical Safety Guardrails & Ethics
* **Explainable AI (XAI):** Doctors are provided with an "AI X-Ray" showing the weight of each symptom that led to the ML prediction.
* **Confidence Thresholding:** If the AI confidence falls below 35%, the system automatically flags the result as "Inconclusive Data" and warns the doctor, preventing misdiagnosis from vague inputs.
* **Patient View Filtering:** Raw, Western disease labels are masked on the Patient Dashboard. Instead, patients see comforting, actionable advice.

---

## 🛠️ System Architecture & Tech Stack

**Frontend:**
* React.js (Vite)
* Tailwind CSS (Styling)
* Framer Motion (Fluid Animations)
* Lucide React (Iconography)

**Backend & Cloud:**
* FastAPI (Serverless API layer on Vercel)
* Google Firebase Authentication (Email/Password & Single Sign-On via Google)
* Google Firestore (NoSQL Database for Users, Patient Records, and Diaries)
* GitHub Actions (CI/CD Automated Linting & Build pipelines)

**Artificial Intelligence:**
* Scikit-Learn (Random Forest Classifier + SMOTE)
* Google Gemini 2.5 Pro API (Generative LLM)

---

## ⚙️ How to Deploy & Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/arogya-ai.git
cd arogya-ai
```

### 2. Start the Backend API (FastAPI)
Ensure you have Python installed, then navigate to the root directory:

```bash
pip install -r requirements.txt
uvicorn api.index:app --reload
```
The FastAPI server will start running on `http://127.0.0.1:8000`. Test the endpoint at `/api/health`.

### 3. Start the React Frontend
Open a new terminal window, navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```
The React app will start running on `http://localhost:5173`. Proxies are mapped so your frontend gracefully calls the backend on `localhost:8000/api`.

### 4. Deployment (Vercel & Render)
Due to Vercel's 500MB serverless limit, this architecture is split for maximum free-tier performance:
1. **Frontend (Vercel):** Connect your GitHub repo to Vercel. Vercel automatically detects Vite. Add an environment variable `VITE_API_URL` pointing to your Render backend URL.
2. **Backend (Render):** Connect your repository to Render as a "Blueprint" using the included `render.yaml`. Add your `GEMINI_API_KEY` to the environment variables. Render will automatically spin up the Python FastAPI server.

---

## 🚀 Future Roadmap (What's Next?)
ArogyaAI is built to scale. Future iterations of this platform will focus on continuous learning and broader accessibility:

1. **Continuous AI Learning (Feedback Loop):** Allow doctors to "Accept" or "Correct" the AI's diagnosis.
2. **Wearable IoT Integration:** Connect with smartwatches (Apple Watch / Fitbit APIs) to automatically pull real-time vitals.
3. **Multilingual Support for Rural Access:** Integrate translation APIs so patients in rural India can log symptoms in regional languages.
4. **Telemedicine & Appointment Booking:** Add a scheduling system where doctors can trigger video calls.

---

*👨‍💻 Academic Integrity & Acknowledgements: This project was developed to demonstrate full-stack software engineering, ethical AI implementation, and modern cloud database architecture. ArogyaAI is a prototype Clinical Decision Support System. It is designed to assist, not replace, licensed medical professionals.*

🏃 Lifestyle Advice
- Practice gentle yoga and stretching exercises daily to maintain flexibility
- Keep joints warm, especially during cold weather
- Apply warm sesame oil massage to affected joints before bathing
- Maintain regular sleep schedule (sleep before 10 PM, wake before 6 AM)
- Stay active but avoid overexertion
- Practice stress management through meditation and pranayama

🌿 Home Remedies & Precautions
- Drink warm water with ginger throughout the day
- Apply warm sesame or castor oil to painful joints
- Use heating pads or warm compresses on affected areas
- Take turmeric milk (1 tsp turmeric in warm milk) before bed
- Gentle massage with warm oils improves circulation
- Epsom salt bath can provide relief

👤 How Treatment Affects Your Body Type
These Ayurvedic treatments specifically address Vata imbalance by providing warmth, 
lubrication, and nourishment to your joints. The warm, oily therapies counteract the 
cold, dry nature of aggravated Vata, helping restore balance and mobility. Regular 
practice will strengthen your tissues, reduce inflammation, and improve overall joint health.

⚠️ Important Note: This is a complementary Ayurvedic approach. For severe arthritis, 
persistent pain, or worsening symptoms, please consult with a qualified healthcare 
professional or rheumatologist for comprehensive medical evaluation and treatment.

---
💡 This analysis combines ML prediction (96.50% confidence) with traditional Ayurvedic 
wisdom to provide personalized recommendations based on your unique constitution and symptoms.
```

## System Architecture

1. **Data Processing**: TF-IDF vectorization of symptoms (807 features)
2. **Feature Engineering**: 12 basic health features + 807 symptom features = 819 total features
3. **Class Balancing**: SMOTE applied for balanced training dataset (4201 → 20748 samples)
4. **Model Training**: Random Forest/Logistic Regression/SVM with ensemble approach
5. **ML Prediction**: Initial disease prediction using trained Random Forest model
6. **LLM Analysis**: Advanced contextual analysis considering all symptoms and lifestyle factors
7. **Ayurvedic Integration**: Comprehensive traditional medicine database with personalized recommendations
8. **Confidence Calibration**: Intelligent assessment of prediction confidence based on symptom patterns

## Model Performance

- **Random Forest Accuracy**: 1.0000 (100%)
- **Logistic Regression Accuracy**: 0.9964 (99.64%)
- **SVM Accuracy**: 0.9417 (94.17%)
- **Feature Set**: 819 features (12 basic + 807 TF-IDF)
- **Training Dataset**: 4,201 samples across 399 diseases
- **SMOTE Augmentation**: 20,748 balanced samples
- **Diseases Supported**: 399 disease categories with full Ayurvedic treatment plans

## Supported Diseases

The system supports diagnosis and Ayurvedic treatment recommendations for **399+ diseases** across multiple medical categories:

### Infectious Diseases
- Common Cold
- Influenza (Flu)
- Pneumonia
- Tuberculosis
- Dengue
- Malaria
- Chikungunya
- Chicken Pox
- Measles
- Mumps
- Typhoid
- Hepatitis A, B, C
- HIV/AIDS
- Urinary Tract Infection
- Skin Infection
- Fungal Infection
- Cholera
- Diarrhoea

### Metabolic & Endocrine Disorders
- Diabetes
- Hypertension
- Thyroid Disorders (Hyper/Hypothyroidism)
- Obesity
- PCOS (Polycystic Ovary Syndrome)
- Anemia

### Respiratory Conditions
- Asthma
- Bronchitis
- Sinusitis
- Cough
- Fever

### Cardiovascular Diseases
- Heart Disease
- Stroke

### Digestive System Disorders
- Gastritis
- Gastroenteritis
- Constipation
- Appendicitis
- Peptic Ulcer
- Jaundice
- Cirrhosis
- Fatty Liver
- Gallstones
- Kidney Stones

### Neurological & Mental Health
- Migraine
- Headache
- Depression
- Anxiety
- Insomnia
- Epilepsy
- Alzheimer Disease
- Parkinson Disease
- Meningitis
- Encephalitis

### Musculoskeletal Disorders
- Arthritis
- Cervical Spondylosis
- Disc Prolapse
- Rheumatoid Arthritis
- Gout
- Osteoporosis
- Paralysis

### Skin & Hair Conditions
- Acne
- Allergy
- Eczema
- Psoriasis
- Dandruff
- Hair Loss
- Vitiligo

### Eye, Ear & Dental Issues
- Conjunctivitis
- Glaucoma
- Cataract
- Hearing Loss
- Tinnitus
- Vertigo
- Toothache
- Gum Disease
- Mouth Ulcer

### Reproductive & Urogenital Health
- Erectile Dysfunction
- Infertility
- Menstrual Disorders
- Endometriosis
- Menopause
- Prostate Enlargement
- Overactive Bladder
- Urinary Incontinence

### Cancer Types
- Breast Cancer
- Various other cancer types

### Other Conditions
- Cancer (various types)
- Sleep Apnea
- Chronic Fatigue Syndrome
- Fibromyalgia
- Autoimmune Disorders
- Lupus
- Multiple Sclerosis
- Carpal Tunnel Syndrome
- Tennis Elbow
- Plantar Fasciitis
- Chronic Pain Syndromes

*The complete system supports 399+ diseases in total, with comprehensive Ayurvedic treatment recommendations for each condition.*

## Usage Modes

### 1. Demo Mode (Default)
Runs sample predictions with pre-defined test cases demonstrating the system capabilities.

### 2. Interactive Mode
Collects user symptoms and health information through an intuitive questionnaire:
- Symptoms input
- Age, height, weight
- Gender and age group
- Enhanced dosha selection with detailed descriptions
- Lifestyle factors (food habits, medication, allergies)
- Environmental factors (season, weather)

### 3. API Integration (Ready)
The system is designed to be easily integrated into web applications or APIs.

### 4. Offline Mode 🔌
**NEW**: The system now works completely offline when internet/LLM is unavailable:
- Automatic detection of LLM availability
- Graceful fallback to local Ayurvedic database
- Same ML prediction accuracy (100%)
- Comprehensive offline recommendations from 4,201+ disease database
- No API key required for basic operation
- Perfect for privacy-sensitive deployments

See [FALLBACK_MECHANISM.md](FALLBACK_MECHANISM.md) for detailed documentation.



## Technical Implementation

- **ML Framework**: Scikit-learn
- **Text Processing**: TF-IDF Vectorization
- **Class Balancing**: SMOTE (Synthetic Minority Oversampling)
- **Feature Scaling**: StandardScaler
- **Model Persistence**: Joblib
- **Data Processing**: Pandas, NumPy
- **Confidence Calibration**: Advanced prediction gap analysis

## File Structure

```
├── train_model.py           # Model training script
├── arogya_predict.py        # Main prediction system with interactive mode
├── disease_prediction_system.py  # Alternative comprehensive implementation
├── demo.py                  # Detailed system demonstration
├── test_fallback.py         # Fallback mechanism testing script
├── requirements.txt         # Python dependencies
├── random_forest_model.pkl  # Trained model (generated)
├── enhanced_ayurvedic_treatment_dataset.csv  # Comprehensive Ayurvedic treatment database (4,201+ diseases)
├── FALLBACK_MECHANISM.md    # Detailed offline mode documentation
└── AyurCore.ipynb          # Original research notebook
```

## Enhanced Ayurvedic Knowledge Base

The system includes an expanded database of traditional Ayurvedic treatments with:
- 50+ traditional herbs with Sanskrit and English names
- 30+ therapeutic treatments and procedures
- Dosha-specific recommendations for Vata, Pitta, and Kapha constitutions
- Personalized dietary guidelines
- Treatment effects explanation for different body types
- Advanced disease-to-treatment mapping with fallback options
- Dynamic treatment personalization based on body constitution

## Future Enhancements

- ~~Integration with real medical datasets~~ ✅ **DONE**
- ~~Enhanced confidence scoring~~ ✅ **DONE**
- ~~Interactive assessment mode~~ ✅ **DONE**
- ~~Top-5 predictions~~ ✅ **DONE**
- ~~Comprehensive dosha selection~~ ✅ **DONE**
- Web interface for easier access
- Mobile application
- Multi-language support
- Advanced NLP for symptom processing
- Telemedicine integration
- User authentication and history
- Dashboard for healthcare providers

## Disclaimer

This system is for educational and research purposes. Always consult with qualified healthcare professionals for medical advice and treatment.

---

**Stay healthy with the wisdom of Ayurveda! 🌿**
