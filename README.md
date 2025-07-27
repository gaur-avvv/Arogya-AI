# ArogyaAI - Hybrid AI-Powered Healthcare Platform

ArogyaAI is a multilingual, AI-powered health assistant that combines modern allopathic diagnostics with ancient Ayurvedic wisdom. It aims to provide accessible, personalized, and preventive healthcare to communities through intelligent automation, voice input, smart reminders, and integration with wearables.


---

🚀 Features

✅ Doctor Module

🎙️ Voice-to-text prescription generation

📄 Automatic prescription formatting

🔄 Upload or update patient health history

🗣️ Indian language support (Hindi, Tamil, etc.)


✅ Patient App

🧠 AI Symptom Checker (text/voice input)

🕉️ Ayurvedic Dosha Quiz & lifestyle guidance

💊 Medicine reminders based on prescription

📸 Proof-of-intake (camera/voice/smartwatch)

📂 Upload old prescriptions/reports


✅ Smartwatch & IoT Integration

⌚ Monitor vitals (heart rate, sleep, activity)

⚠️ Health anomaly alerts

🧘 Personalized suggestions based on data trends


✅ Health Comparison & Choice

🔬 Side-by-side Allopathic vs. Ayurvedic suggestions

📚 Educational snippets on treatments



---

🧰 Tech Stack

🧠 Model: LLaMA 2 / DeepSeek (fine-tuned via HuggingFace AutoTrain)

🗣️ NLP: Whisper, Google Speech-to-Text (voice recognition)

🌐 Frontend: ReactJS, HTML5, TailwindCSS

🔧 Backend: Flask / FastAPI (Python)

🛢️ Database: MongoDB / PostgreSQL

📱 Mobile Companion: React Native (planned)

☁️ Cloud Deployment: Azure (optional), Colab/Spaces for demo



---

📁 Folder Structure

ArogyaAI/
│
├── backend/             # Flask/FastAPI APIs for AI, prescription, reminders
├── frontend/            # ReactJS interface
├── models/              # Finetuned HuggingFace models & configs
├── smartwatch_module/   # Integration with wearable data APIs
├── datasets/            # Custom medical training datasets (JSON, XML)
├── docs/                # PDFs, references, Ayurveda literature
├── mobile_app/          # React Native structure (coming soon)
└── README.md


---

✅ How to Run (Locally)

1. Clone the repo:



git clone https://github.com/gaur-avvv/arogyaai.git
cd arogyaai

2. Install backend dependencies:



cd backend
pip install -r requirements.txt

3. Start the API server:



uvicorn main:app --reload

4. Launch frontend:



cd frontend
npm install && npm start


---

🙌 Contributors

Sakshi Singh – Team Lead, AI Developer

[Add names here]



---

📜 License

This project is open-source under the MIT License.


---

🌟 Acknowledgements

Built for the GCYLP Hackathon. Supported by Microsoft Azure, Hugging Face, and the open-source community.



Here's your complete ArogyaAI project README file ready for GitHub. It includes all your updated features and tech stack. You can now use this as the main README.md when uploading your project to GitHub.

Would you like me to help generate the backend, frontend, or dataset folder boilerplates next?

