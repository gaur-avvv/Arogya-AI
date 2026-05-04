import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyBNriDLDKB7uR1ougWyKb7UGxyPlQu2d4o",
  authDomain: "arogyaai-cloud.firebaseapp.com",
  projectId: "arogyaai-cloud",
  storageBucket: "arogyaai-cloud.firebasestorage.app",
  messagingSenderId: "773413352758",
  appId: "1:773413352758:web:8b5246812f25b062842f97",
  measurementId: "G-VKHCR78XXT",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Export the Authentication and Database modules so our React app can use them
export const auth = getAuth(app);
export const db = getFirestore(app);
