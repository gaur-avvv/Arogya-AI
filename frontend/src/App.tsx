import React, { useState, useEffect } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  useLocation,
  useNavigate,
  Navigate,
} from "react-router-dom";
import {
  Activity,
  ChevronRight,
  BrainCircuit,
  Leaf,
  CheckCircle2,
  Mail,
  Lock,
  LogIn,
  User,
  LayoutDashboard,
  Settings,
  HelpCircle,
  Users,
  Key,
  Shield,
  LogOut,
  Menu,
  X,
  HeartPulse,
  PlusCircle,
  Calendar,
  Building,
  Save,
  ClipboardCheck,
  Wind,
  ActivitySquare,
  MessageSquare,
  PieChart,
  Thermometer,
  AlertCircle,
  Sparkles,
  Download,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// --- FIREBASE CLOUD IMPORTS ---
import { auth, db } from "./firebase";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  GoogleAuthProvider,
  signInWithPopup,
} from "firebase/auth";
import type { User as FirebaseUser } from "firebase/auth";
import {
  collection,
  addDoc,
  getDocs,
  query,
  where,
  serverTimestamp,
  doc,
  setDoc,
  getDoc,
} from "firebase/firestore";

// --- INTERFACES ---
interface FormData {
  name: string;
  age: string;
  gender: string;
  height: string;
  weight: string;
  dosha: string;
  season: string;
  symptoms: string;
}

interface AnalysisResult {
  prediction: string;
  confidence: number;
  xai_breakdown: { symptom: string; weight: number }[];
  indicators: { label: string; score: string }[];
  reasoning: string;
  herbs: { name: string; benefit: string }[];
  lifestyle: string[];
  recommendation?: string;
}

interface UserData {
  role: string;
  clinicId: string;
  email: string;
}

// --- COLLAPSIBLE SIDEBAR COMPONENT ---
function Sidebar({
  user,
  userData,
  isOpen,
  setIsOpen,
}: {
  user: FirebaseUser | null;
  userData: UserData | null;
  isOpen: boolean;
  setIsOpen: (val: boolean) => void;
}) {
  const location = useLocation();
  const userRole = userData?.role;

  const navItems =
    userRole === "doctor"
      ? [
          {
            name: "Clinic Dashboard",
            path: "/",
            icon: <LayoutDashboard size={22} />,
          },
          {
            name: "AI Diagnostic",
            path: "/diagnose",
            icon: <BrainCircuit size={22} />,
          },
          {
            name: "Patient Records",
            path: "/patients",
            icon: <Users size={22} />,
          },
          {
            name: "Clinic Profile",
            path: "/profile",
            icon: <Settings size={22} />,
          },
          {
            name: "Help Center",
            path: "/help",
            icon: <HelpCircle size={22} />,
          },
        ]
      : [
          { name: "My Health", path: "/", icon: <HeartPulse size={22} /> },
          {
            name: "Symptom Logger",
            path: "/checkup",
            icon: <PlusCircle size={22} />,
          },
          {
            name: "Profile Settings",
            path: "/profile",
            icon: <Settings size={22} />,
          },
        ];

  return (
    <aside
      className={`w-72 bg-slate-950 text-slate-300 min-h-screen p-6 flex flex-col no-print fixed z-[100] transition-transform duration-300 ease-in-out ${isOpen ? "translate-x-0" : "-translate-x-full"}`}
    >
      <div className="flex items-center justify-between mb-12 mt-4 px-2">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-emerald-500 to-teal-700 p-2.5 rounded-xl shadow-lg shadow-emerald-900/50">
            <Leaf className="text-white w-6 h-6" />
          </div>
          <span className="text-2xl font-black tracking-tight text-white">
            Arogya<span className="text-emerald-400">AI</span>
          </span>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="md:hidden text-slate-400 hover:text-white transition-colors"
        >
          <X size={24} />
        </button>
      </div>

      <div className="mb-6 px-4 text-xs font-black uppercase tracking-widest text-slate-500">
        {userRole === "doctor" ? "Practitioner Portal" : "Patient Portal"}
      </div>

      <nav className="space-y-2 flex-1">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.name}
              to={item.path}
              onClick={() => window.innerWidth < 768 && setIsOpen(false)}
              className={`flex items-center gap-4 px-4 py-4 rounded-2xl font-bold transition-all ${isActive ? "bg-emerald-500/10 text-emerald-400" : "hover:bg-slate-900 hover:text-white"}`}
            >
              {item.icon} {item.name}
              {isActive && (
                <motion.div
                  layoutId="active-pill"
                  className="w-1.5 h-8 bg-emerald-500 absolute left-0 rounded-r-full"
                />
              )}
            </Link>
          );
        })}
      </nav>

      <div
        className="mt-auto bg-slate-900 p-4 rounded-2xl flex items-center gap-4 border border-slate-800 relative group cursor-pointer"
        onClick={() => signOut(auth)}
      >
        <div className="w-10 h-10 bg-emerald-900 rounded-full flex items-center justify-center text-emerald-400 font-black">
          <User size={20} />
        </div>
        <div className="overflow-hidden">
          <p className="text-white font-bold text-sm truncate">
            {user?.email?.split("@")[0] || "User"}
          </p>
          <p className="text-slate-500 text-xs font-medium group-hover:hidden capitalize">
            {userRole}
          </p>
          <p className="text-red-400 text-xs font-bold hidden group-hover:flex items-center gap-1">
            <LogOut size={12} /> Sign Out
          </p>
        </div>
      </div>
    </aside>
  );
}

// --- PATIENT SYMPTOM LOGGER ---
function PatientCheckup({
  user,
  userData,
}: {
  user: FirebaseUser | null;
  userData: UserData | null;
}) {
  const [symptoms, setSymptoms] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSave = async () => {
    if (!symptoms.trim() || !userData) return;
    setIsSaving(true);
    try {
      await addDoc(collection(db, "patient_logs"), {
        userId: user?.uid,
        email: user?.email,
        symptoms: symptoms,
        clinicId: userData.clinicId,
        createdAt: serverTimestamp(),
      });
      setSavedSuccess(true);
      setSymptoms("");
      setTimeout(() => setSavedSuccess(false), 4000);
    } catch (error) {
      console.error("Error saving log:", error);
      alert("Failed to save. Please check your connection.");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto space-y-8 p-10"
    >
      <div>
        <h1 className="text-4xl font-black text-slate-950 tracking-tighter mb-2">
          Symptom Logger
        </h1>
        <p className="text-slate-500 font-medium text-lg">
          Securely record how you are feeling today.
        </p>
      </div>

      <div className="bg-white p-10 md:p-12 rounded-[3rem] border border-slate-200/60 shadow-sm space-y-8">
        <div className="flex items-center gap-4 bg-blue-50 text-blue-800 p-6 rounded-2xl border border-blue-100">
          <Shield className="w-8 h-8 flex-shrink-0" />
          <p className="text-sm font-medium">
            Your logs are securely encrypted and sent directly to Clinic ID{" "}
            <strong>{userData?.clinicId}</strong>. Only your practitioner can
            access this data.
          </p>
        </div>

        <div className="space-y-4">
          <label className="text-xl font-black text-slate-900">
            How are you feeling?
          </label>
          <p className="text-slate-500 font-medium text-sm">
            Please describe any pain, discomfort, sleep issues, or digestive
            changes you have noticed recently.
          </p>
          <textarea
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            className="w-full p-6 h-64 text-lg font-medium rounded-[2rem] border-2 border-slate-200/60 bg-slate-50 focus:ring-4 focus:ring-emerald-200 outline-none resize-none shadow-inner"
            placeholder="e.g. I have had a mild headache for two days and my digestion feels sluggish..."
          ></textarea>
        </div>

        <button
          onClick={handleSave}
          disabled={!symptoms.trim() || isSaving || savedSuccess}
          className={`w-full py-5 rounded-2xl font-black text-xl flex items-center justify-center gap-3 transition-all ${savedSuccess ? "bg-teal-600 text-white" : "bg-slate-950 text-white hover:bg-slate-800 disabled:opacity-50"}`}
        >
          {isSaving ? (
            "Saving securely..."
          ) : savedSuccess ? (
            <>
              <CheckCircle2 size={24} /> Saved to Health Diary
            </>
          ) : (
            <>
              <Save size={24} /> Submit to Diary
            </>
          )}
        </button>
      </div>
    </motion.div>
  );
}

// --- UPDATED: PATIENT DASHBOARD COMPONENT (SAFE TITLES) ---
function PatientDashboard({
  user,
  userData,
}: {
  user: FirebaseUser | null;
  userData: UserData | null;
}) {
  const [prescriptions, setPrescriptions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const patientName = user?.email?.split("@")[0] || "Patient";

  useEffect(() => {
    const fetchMyPrescriptions = async () => {
      if (!userData?.clinicId || !user?.email) return;
      try {
        const q = query(
          collection(db, "patients"),
          where("clinicId", "==", userData.clinicId),
        );
        const snap = await getDocs(q);

        let myDocs = snap.docs.map((doc) => ({
          id: doc.id,
          ...(doc.data() as any),
        }));
        myDocs = myDocs.filter((doc) =>
          doc.name.toLowerCase().includes(patientName.toLowerCase()),
        );
        myDocs.sort(
          (a, b) =>
            (b.createdAt?.toMillis() || 0) - (a.createdAt?.toMillis() || 0),
        );

        setPrescriptions(myDocs);
      } catch (error) {
        console.error("Error fetching prescriptions:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchMyPrescriptions();
  }, [userData, user]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-7xl mx-auto space-y-8 p-10"
    >
      <div>
        <h1 className="text-4xl font-black text-slate-950 tracking-tighter mb-2">
          Hello, {patientName}.
        </h1>
        <p className="text-slate-500 font-medium text-lg">
          Welcome to your personal Ayurvedic health portal.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-emerald-500 to-teal-600 p-10 rounded-[2rem] text-white shadow-lg relative overflow-hidden">
          <div className="absolute -right-4 -bottom-4 opacity-20">
            <HeartPulse size={150} />
          </div>
          <h3 className="text-2xl font-black mb-4 relative z-10">
            Start a Quick Checkup
          </h3>
          <p className="text-emerald-100 font-medium mb-8 relative z-10 max-w-sm">
            Not feeling well? Log your symptoms to share securely with your
            Ayurvedic practitioner.
          </p>
          <Link
            to="/checkup"
            className="bg-white text-teal-700 px-6 py-3 rounded-full font-black text-sm relative z-10 shadow-md hover:bg-slate-50 inline-block transition-colors"
          >
            Log Symptoms
          </Link>
        </div>

        <div className="bg-white p-10 rounded-[2rem] border border-slate-200/60 shadow-sm flex flex-col relative">
          <div className="flex items-center gap-3 mb-6 border-b border-slate-100 pb-4">
            <ClipboardCheck className="text-blue-500" size={24} />
            <h3 className="text-xl font-black text-slate-900">
              Recent Practitioner Advice
            </h3>
          </div>

          <div className="flex-1 overflow-y-auto max-h-64 pr-2">
            {loading ? (
              <p className="text-slate-400 font-bold text-sm text-center mt-10">
                Checking clinic records...
              </p>
            ) : prescriptions.length === 0 ? (
              <div className="flex flex-col justify-center items-center h-full text-center opacity-50 mt-4">
                <Leaf size={40} className="mb-3 text-slate-400" />
                <p className="text-slate-500 font-bold text-sm">
                  No recent herbal protocols uploaded by your doctor yet.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {prescriptions.map((presc) => (
                  <div
                    key={presc.id}
                    className="bg-slate-50 p-5 rounded-2xl border border-slate-100"
                  >
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-xs font-black text-slate-400 uppercase tracking-widest">
                        {presc.createdAt
                          ? presc.createdAt.toDate().toLocaleDateString()
                          : "Recent"}
                      </span>
                      <span className="text-xs font-bold bg-blue-100 text-blue-700 px-2 py-1 rounded-md">
                        Dr. Review
                      </span>
                    </div>
                    {/* SAFETY FIX: We no longer show the ML disease. We show a comforting Ayurvedic title based on their dosha */}
                    <p className="text-sm font-black text-slate-800 mb-3">
                      Personalized {presc.dosha.split(" ")[0]} Balancing
                      Protocol
                    </p>
                    <div className="space-y-2">
                      <p className="text-xs font-black text-emerald-600 flex gap-1">
                        <Leaf size={14} /> Recommended Herbs:
                      </p>
                      <p className="text-sm font-medium text-slate-600 line-clamp-2">
                        This protocol focuses on restoring harmony to your
                        system. Please refer to your practitioner for detailed
                        dosage.
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

// --- GLOBAL DOCTOR DASHBOARD COMPONENT ---
function GlobalDashboard() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-7xl mx-auto space-y-8 p-10"
    >
      <div>
        <h1 className="text-4xl font-black text-slate-950 tracking-tighter mb-2">
          Welcome back, Doctor.
        </h1>
        <p className="text-slate-500 font-medium text-lg">
          Here is your clinic's AI diagnostic overview for this week.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-8 rounded-[2rem] border border-slate-200/60 shadow-sm">
          <div className="w-12 h-12 bg-emerald-100 text-emerald-600 rounded-2xl flex items-center justify-center mb-6">
            <Activity size={24} />
          </div>
          <p className="text-slate-500 font-bold uppercase tracking-widest text-xs mb-1">
            AI Diagnostics Run
          </p>
          <p className="text-5xl font-black text-slate-950">142</p>
        </div>
        <div className="bg-white p-8 rounded-[2rem] border border-slate-200/60 shadow-sm">
          <div className="w-12 h-12 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mb-6">
            <BrainCircuit size={24} />
          </div>
          <p className="text-slate-500 font-bold uppercase tracking-widest text-xs mb-1">
            Average ML Confidence
          </p>
          <p className="text-5xl font-black text-slate-950">
            94.2<span className="text-2xl">%</span>
          </p>
        </div>
        <div className="bg-white p-8 rounded-[2rem] border border-slate-200/60 shadow-sm">
          <div className="w-12 h-12 bg-orange-100 text-orange-600 rounded-2xl flex items-center justify-center mb-6">
            <Wind size={24} />
          </div>
          <p className="text-slate-500 font-bold uppercase tracking-widest text-xs mb-1">
            Dominant Clinic Dosha
          </p>
          <p className="text-5xl font-black text-slate-950">Vata</p>
        </div>
      </div>

      <div className="bg-slate-950 rounded-[3rem] p-12 text-white relative overflow-hidden mt-8">
        <div className="absolute right-0 top-0 opacity-10">
          <ActivitySquare size={300} className="-mr-10 -mt-10" />
        </div>
        <h2 className="text-3xl font-black mb-4">
          Start a new patient analysis
        </h2>
        <p className="text-slate-400 mb-8 max-w-xl text-lg">
          Use the hybrid AI engine to generate detailed predictions, holistic
          protocols, and save records securely to the cloud.
        </p>
        <Link
          to="/diagnose"
          className="bg-emerald-500 text-white px-8 py-4 rounded-full font-black text-lg inline-flex items-center gap-2 hover:bg-emerald-400 transition-colors"
        >
          Initialize AI Tool <ChevronRight size={20} />
        </Link>
      </div>
    </motion.div>
  );
}

// --- CLOUD-CONNECTED PATIENT RECORDS ---
function PatientRecords({ userData }: { userData: UserData | null }) {
  const [activeTab, setActiveTab] = useState<"diagnostics" | "logs">(
    "diagnostics",
  );
  const [patients, setPatients] = useState<any[]>([]);
  const [patientLogs, setPatientLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      if (!userData?.clinicId) return;
      try {
        const qPatients = query(
          collection(db, "patients"),
          where("clinicId", "==", userData.clinicId),
        );
        const snapPatients = await getDocs(qPatients);
        let pts = snapPatients.docs.map((doc) => ({
          id: doc.id,
          ...doc.data(),
        }));
        pts.sort(
          (a: any, b: any) =>
            (b.createdAt?.toMillis() || 0) - (a.createdAt?.toMillis() || 0),
        );
        setPatients(pts);

        const qLogs = query(
          collection(db, "patient_logs"),
          where("clinicId", "==", userData.clinicId),
        );
        const snapLogs = await getDocs(qLogs);
        let logs = snapLogs.docs.map((doc) => ({ id: doc.id, ...doc.data() }));
        logs.sort(
          (a: any, b: any) =>
            (b.createdAt?.toMillis() || 0) - (a.createdAt?.toMillis() || 0),
        );
        setPatientLogs(logs);
      } catch (error) {
        console.error("Error fetching records:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [userData]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-7xl mx-auto space-y-8 p-10"
    >
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-4xl font-black text-slate-950 tracking-tighter mb-2">
            Patient Records
          </h1>
          <p className="text-slate-500 font-medium text-lg">
            Secure clinical history for Clinic ID:{" "}
            <strong>{userData?.clinicId}</strong>.
          </p>
        </div>
      </div>

      <div className="flex gap-4 border-b-2 border-slate-200/60 pb-4">
        <button
          onClick={() => setActiveTab("diagnostics")}
          className={`px-6 py-3 rounded-full font-bold text-sm flex items-center gap-2 transition-all ${activeTab === "diagnostics" ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500 hover:bg-slate-200"}`}
        >
          <ActivitySquare size={18} /> Clinic Diagnostics
        </button>
        <button
          onClick={() => setActiveTab("logs")}
          className={`px-6 py-3 rounded-full font-bold text-sm flex items-center gap-2 transition-all ${activeTab === "logs" ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500 hover:bg-slate-200"}`}
        >
          <MessageSquare size={18} /> Patient Health Diaries
        </button>
      </div>

      <div className="bg-white rounded-[2rem] border border-slate-200/60 shadow-sm overflow-hidden overflow-x-auto">
        {loading ? (
          <div className="p-10 text-center text-slate-500 font-bold">
            Loading secure records from cloud...
          </div>
        ) : activeTab === "diagnostics" ? (
          patients.length === 0 ? (
            <div className="p-10 text-center text-slate-500 font-bold">
              No AI diagnostic records found for this clinic.
            </div>
          ) : (
            <table className="w-full text-left min-w-[800px]">
              <thead className="bg-slate-50 border-b border-slate-200/60">
                <tr>
                  <th className="p-6 font-black text-slate-400 text-sm uppercase tracking-widest">
                    Name
                  </th>
                  <th className="p-6 font-black text-slate-400 text-sm uppercase tracking-widest">
                    Profile
                  </th>
                  <th className="p-6 font-black text-slate-400 text-sm uppercase tracking-widest">
                    Prakriti (Dosha)
                  </th>
                  <th className="p-6 font-black text-slate-400 text-sm uppercase tracking-widest">
                    ML Diagnosis
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {patients.map((pt, i) => (
                  <tr
                    key={i}
                    className="hover:bg-slate-50 transition-colors cursor-pointer"
                  >
                    <td className="p-6 font-black text-slate-900">{pt.name}</td>
                    <td className="p-6 font-bold text-slate-600">
                      {pt.age}y / {pt.gender}
                    </td>
                    <td className="p-6">
                      <span className="bg-slate-100 text-slate-700 px-3 py-1 rounded-full text-xs font-bold border border-slate-200">
                        {pt.dosha.split(" ")[0]}
                      </span>
                    </td>
                    <td className="p-6 font-bold text-emerald-600">
                      {pt.diagnosis}{" "}
                      <span className="text-slate-400 text-xs ml-2">
                        ({pt.confidence}%)
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        ) : patientLogs.length === 0 ? (
          <div className="p-10 text-center text-slate-500 font-bold">
            No patient diaries have been submitted yet.
          </div>
        ) : (
          <div className="p-6 space-y-4">
            {patientLogs.map((log, i) => (
              <div
                key={i}
                className="bg-slate-50 border border-slate-100 p-6 rounded-3xl flex flex-col md:flex-row gap-6"
              >
                <div className="flex-shrink-0 w-12 h-12 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-black">
                  {log.email?.charAt(0).toUpperCase() || "P"}
                </div>
                <div className="flex-1 space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="font-black text-slate-900">{log.email}</h4>
                    <span className="text-xs font-bold text-slate-400 flex items-center gap-1 bg-white px-3 py-1 rounded-full border border-slate-200">
                      <Calendar size={12} />
                      {log.createdAt
                        ? log.createdAt.toDate().toLocaleDateString()
                        : "Just now"}
                    </span>
                  </div>
                  <p className="text-slate-600 font-medium leading-relaxed bg-white p-4 rounded-2xl border border-slate-100 shadow-sm">
                    "{log.symptoms}"
                  </p>

                  <button
                    onClick={() =>
                      navigate("/diagnose", {
                        state: {
                          prefillSymptoms: log.symptoms,
                          prefillName: log.email?.split("@")[0],
                        },
                      })
                    }
                    className="mt-4 bg-emerald-100 text-emerald-700 px-5 py-2.5 rounded-xl font-bold text-sm flex items-center gap-2 hover:bg-emerald-200 transition-colors inline-flex"
                  >
                    <BrainCircuit size={18} /> Analyze with AI
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// --- PROFILE SETTINGS COMPONENT ---
function ProfileSettings({
  user,
  userData,
}: {
  user: FirebaseUser | null;
  userData: UserData | null;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto space-y-8 p-10"
    >
      <div>
        <h1 className="text-4xl font-black text-slate-950 tracking-tighter mb-2">
          Account Settings
        </h1>
        <p className="text-slate-500 font-medium text-lg">
          Manage your {userData?.role} profile details.
        </p>
      </div>

      <div className="bg-white p-10 rounded-[2rem] border border-slate-200/60 shadow-sm space-y-8">
        <div className="flex items-center gap-6 pb-8 border-b border-slate-100">
          <div className="w-24 h-24 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center font-black text-3xl">
            {user?.email?.charAt(0).toUpperCase() || "U"}
          </div>
          <div>
            <h3 className="text-2xl font-black text-slate-900">
              {user?.email || "User"}
            </h3>
            <p className="text-slate-500 font-bold capitalize">
              {userData?.role} Account
            </p>
          </div>
        </div>

        {userData?.role === "doctor" && (
          <div className="bg-blue-50 border border-blue-200 p-6 rounded-2xl flex items-center justify-between">
            <div>
              <h4 className="font-black text-blue-900 flex items-center gap-2 mb-1">
                <Building size={20} /> Your Clinic ID
              </h4>
              <p className="text-blue-800 text-sm font-medium">
                Give this code to your patients so they can link their accounts
                to your clinic.
              </p>
            </div>
            <div className="bg-white px-6 py-3 rounded-xl border border-blue-200 font-mono font-black text-2xl text-blue-600 tracking-widest">
              {userData.clinicId}
            </div>
          </div>
        )}

        {userData?.role === "patient" && (
          <div className="bg-emerald-50 border border-emerald-200 p-6 rounded-2xl">
            <h4 className="font-black text-emerald-900 flex items-center gap-2 mb-1">
              <Shield size={20} /> Privacy & Data Sharing
            </h4>
            <p className="text-emerald-800 text-sm font-medium">
              Your health data is securely encrypted and shared exclusively with
              Clinic ID: <strong>{userData.clinicId}</strong>.
            </p>
          </div>
        )}

        {userData?.role === "doctor" && (
          <div className="space-y-6 pt-4">
            <h4 className="text-xl font-black text-slate-900 flex items-center gap-2">
              <Key className="text-emerald-500" /> API Security Gateway
            </h4>
            <p className="text-slate-500 font-medium">
              To bypass free-tier rate limits, input your personal Google Gemini
              API key below. This key is stored locally and never sent to our
              servers.
            </p>

            <div className="space-y-3">
              <label className="text-sm font-black uppercase tracking-widest text-slate-400">
                Gemini LLM API Key
              </label>
              <div className="relative">
                <Key className="absolute left-4 top-4 text-slate-400 w-6 h-6" />
                <input
                  type="password"
                  value={localStorage.getItem("geminiApiKey") || ""}
                  onChange={(e) => {
                    localStorage.setItem("geminiApiKey", e.target.value);
                  }}
                  placeholder="AIzaSy..."
                  className="w-full pl-14 pr-6 py-4 rounded-2xl border-2 border-slate-200 focus:border-emerald-500 outline-none font-bold text-lg font-mono"
                />
              </div>
            </div>

            <div className="space-y-3 mt-4">
              <label className="text-sm font-black uppercase tracking-widest text-slate-400">
                Render API Base URL
              </label>
              <div className="relative">
                <Shield className="absolute left-4 top-4 text-slate-400 w-6 h-6" />
                <input
                  type="url"
                  value={localStorage.getItem("renderUrl") || ""}
                  onChange={(e) => {
                    localStorage.setItem("renderUrl", e.target.value);
                  }}
                  placeholder="https://arogya-ai.onrender.com"
                  className="w-full pl-14 pr-6 py-4 rounded-2xl border-2 border-slate-200 focus:border-emerald-500 outline-none font-bold text-lg font-mono"
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
}

// --- HELP CENTER COMPONENT ---
function HelpCenter() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto space-y-8 p-10"
    >
      <div>
        <h1 className="text-4xl font-black text-slate-950 tracking-tighter mb-2">
          Help Center & Docs
        </h1>
        <p className="text-slate-500 font-medium text-lg">
          Understanding the Clinical Decision Support System.
        </p>
      </div>
      <div className="bg-white p-10 rounded-[2rem] border border-slate-200/60 shadow-sm space-y-8">
        <h4 className="text-xl font-black text-slate-900 border-b border-slate-100 pb-4">
          How does the Hybrid AI work?
        </h4>
        <p className="text-slate-600 font-medium leading-relaxed">
          ArogyaAI uses a <strong>Dual-Engine Architecture</strong>. First,
          patient symptoms are processed by a deterministic{" "}
          <strong>Random Forest Machine Learning model</strong>. Second, a{" "}
          <strong>Generative AI (Gemini 2.5)</strong> model generates holistic
          Ayurvedic context.
        </p>
      </div>
    </motion.div>
  );
}

// --- CLOUD-CONNECTED DIAGNOSTIC TOOL COMPONENT ---
function DiagnosticTool({ userData }: { userData: UserData | null }) {
  const [step, setStep] = useState(1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const [doshaMode, setDoshaMode] = useState<"manual" | "quiz">("manual");
  const [quizAnswers, setQuizAnswers] = useState({
    frame: "",
    digestion: "",
    sleep: "",
  });

  const [formData, setFormData] = useState<FormData>({
    name: "",
    age: "",
    gender: "Male",
    height: "",
    weight: "",
    dosha: "Vata (Air/Space)",
    season: "Summer (Grishma)",
    symptoms: "",
  });

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const renderUrl = localStorage.getItem("renderUrl") || "";
  const geminiApiKey = localStorage.getItem("geminiApiKey") || "";
  const location = useLocation();

  useEffect(() => {
    if (location.state && location.state.prefillSymptoms) {
      setFormData((prev) => ({
        ...prev,
        name: location.state.prefillName || prev.name,
        symptoms: location.state.prefillSymptoms,
      }));
    }
  }, [location.state]);

  const handleInputChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
    >,
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const isStepValid = () => {
    switch (step) {
      case 1:
        return (
          formData.name &&
          formData.age &&
          formData.gender &&
          formData.height &&
          formData.weight
        );
      case 2:
        if (doshaMode === "manual") return formData.dosha && formData.season;
        return (
          quizAnswers.frame &&
          quizAnswers.digestion &&
          quizAnswers.sleep &&
          formData.season
        );
      case 3:
        return formData.symptoms.trim().length > 5;
      default:
        return false;
    }
  };

  const calculateDosha = () => {
    let v = 0,
      p = 0,
      k = 0;
    if (quizAnswers.frame === "Thin/Light") v += 40;
    if (quizAnswers.frame === "Medium/Athletic") p += 40;
    if (quizAnswers.frame === "Heavy/Solid") k += 40;
    if (quizAnswers.digestion === "Irregular/Gas") v += 30;
    if (quizAnswers.digestion === "Strong/Acidic") p += 30;
    if (quizAnswers.digestion === "Slow/Sluggish") k += 30;
    if (quizAnswers.sleep === "Light/Interrupted") v += 30;
    if (quizAnswers.sleep === "Moderate/Sound") p += 30;
    if (quizAnswers.sleep === "Deep/Prolonged") k += 30;
    return `Vata ${v}%, Pitta ${p}%, Kapha ${k}%`;
  };

  const handleNext = () => {
    if (isStepValid()) {
      if (step === 2 && doshaMode === "quiz")
        setFormData((prev) => ({ ...prev, dosha: calculateDosha() }));
      setStep(step + 1);
      setError(null);
    } else {
      setError("Please complete all patient data fields.");
    }
  };

  const runAnalysis = async () => {
    if (!isStepValid()) return;
    setIsAnalyzing(true);
    setError(null);
    setSavedSuccess(false);
    try {
      // Map formData to Backend API structure
      const apiPayload = {
        Symptoms: formData.symptoms,
        Age: parseInt(formData.age),
        Height_cm: parseInt(formData.height || "170"),
        Weight_kg: parseInt(formData.weight || "70"),
        Gender: formData.gender,
        Body_Type_Dosha_Sanskrit: formData.dosha,
        Season: formData.season,
        Food_Habits: "Mixed", 
        Current_Medication: "None",
        Allergies: "None",
        Weather: "Clear"
      };

      // Determine API URL based on environment or user input
      const API_BASE_URL = renderUrl || import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(apiPayload),
      });

      if (!response.ok)
        throw new Error("AI Server unreachable or API limit hit.");

      const data = await response.json();
      
      // If user provided Gemini Key, we can fetch LLM details explicitly
      if (geminiApiKey) {
        try {
          const prompt = `You are an expert Ayurvedic health assistant. Your task is to analyze a user's health data and an initial model prediction, then provide a final, trustworthy, and personalized Ayurvedic diagnosis and plan.

**User's Health Profile:**
- **Symptoms:** ${apiPayload.Symptoms}
- **Age:** ${apiPayload.Age}
- **Gender:** ${apiPayload.Gender}
- **Body Type (Dosha):** ${apiPayload.Body_Type_Dosha_Sanskrit}

**Initial Analysis:**
- **Predicted Condition:** ${data.prediction || data.ml_prediction}
- **Initial Confidence:** ${data.confidence}

Provide a concise, easy to read explanation and a personalized plan featuring herbs and diet based on Ayurveda.`;

          const geminiReq = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${geminiApiKey}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                contents: [{ parts: [{ text: prompt }] }],
              }),
            }
          );
          const geminiRes = await geminiReq.json();
          if (geminiRes.candidates && geminiRes.candidates.length > 0) {
             data.recommendation = geminiRes.candidates[0].content.parts[0].text;
          }
        } catch (e) {
          console.warn("Gemini client-side fetch failed", e);
        }
      }

      setResult(data);
      setShowResult(true);
    } catch (err: any) {
      setError(err.message || "AI Analysis Server unreachable.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const saveToCloud = async () => {
    if (!result || !userData) return;
    setIsSaving(true);
    try {
      // SAFETY GUARDRAIL: Do not save scary diseases if the AI is guessing blindly
      const safeDiagnosis =
        result.confidence < 35
          ? "General Imbalance (Review Required)"
          : result.prediction;

      await addDoc(collection(db, "patients"), {
        name: formData.name,
        age: formData.age,
        gender: formData.gender,
        dosha: formData.dosha,
        symptoms: formData.symptoms,
        diagnosis: safeDiagnosis, // Safely scrubbed diagnosis
        confidence: result.confidence,
        clinicId: userData.clinicId,
        createdAt: serverTimestamp(),
      });
      setSavedSuccess(true);
    } catch (err) {
      console.error("Error saving to cloud:", err);
      alert("Failed to save to cloud database.");
    } finally {
      setIsSaving(false);
    }
  };

  const isLowConfidence = result && result.confidence < 35;

  return (
    <div className="max-w-[1600px] mx-auto p-6 md:p-10 md:pt-4">
      <AnimatePresence mode="wait">
        {!showResult && !isAnalyzing && (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-12"
          >
            {location.state?.prefillSymptoms && step === 1 && (
              <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 p-4 rounded-2xl flex items-center gap-3 mx-6">
                <CheckCircle2 className="flex-shrink-0" size={20} />
                <p className="font-bold text-sm">
                  Patient symptoms have been securely loaded. Please complete
                  the demographics to proceed to the AI Analysis.
                </p>
              </div>
            )}

            <div className="mb-12 space-y-5 px-6">
              <div className="flex justify-between text-xs md:text-sm font-black uppercase tracking-widest text-slate-400">
                <span
                  className={step >= 1 ? "text-emerald-700 font-black" : ""}
                >
                  1. Data
                </span>
                <span
                  className={step >= 2 ? "text-emerald-700 font-black" : ""}
                >
                  2. Context
                </span>
                <span
                  className={step >= 3 ? "text-emerald-700 font-black" : ""}
                >
                  3. Symptoms
                </span>
              </div>
            </div>

            <div className="bg-white rounded-[3rem] shadow-[0_15px_60px_rgb(0,0,0,0.06)] border border-slate-100 p-8 md:p-16">
              {step === 1 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-10"
                >
                  <h1 className="text-4xl md:text-5xl font-black text-slate-950 tracking-tighter">
                    1. Demographics
                  </h1>
                  <div className="space-y-3">
                    <label className="text-lg font-bold text-slate-700">
                      Patient Full Name
                    </label>
                    <input
                      type="text"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className="w-full p-6 text-xl rounded-3xl border-2 border-slate-200/60 bg-slate-50 focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                      placeholder="e.g. Ananya Sharma"
                    />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10">
                    <div className="space-y-3">
                      <label className="text-lg font-bold text-slate-700">
                        Patient Age
                      </label>
                      <input
                        type="number"
                        name="age"
                        value={formData.age}
                        onChange={handleInputChange}
                        className="w-full p-6 text-xl rounded-3xl border-2 border-slate-200/60 bg-slate-50 focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                      />
                    </div>
                    <div className="space-y-3">
                      <label className="text-lg font-bold text-slate-700">
                        Patient Sex
                      </label>
                      <select
                        name="gender"
                        value={formData.gender}
                        onChange={handleInputChange}
                        className="w-full p-6 text-xl rounded-3xl border-2 border-slate-200/60 bg-slate-50 focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                      >
                        <option>Male</option>
                        <option>Female</option>
                        <option>Other</option>
                      </select>
                    </div>
                    <div className="space-y-3">
                      <label className="text-lg font-bold text-slate-700">
                        Height (cm)
                      </label>
                      <input
                        type="number"
                        name="height"
                        value={formData.height}
                        onChange={handleInputChange}
                        className="w-full p-6 text-xl rounded-3xl border-2 border-slate-200/60 bg-slate-50 focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                      />
                    </div>
                    <div className="space-y-3">
                      <label className="text-lg font-bold text-slate-700">
                        Weight (kg)
                      </label>
                      <input
                        type="number"
                        name="weight"
                        value={formData.weight}
                        onChange={handleInputChange}
                        className="w-full p-6 text-xl rounded-3xl border-2 border-slate-200/60 bg-slate-50 focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                      />
                    </div>
                  </div>
                </motion.div>
              )}

              {step === 2 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-10"
                >
                  <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <h1 className="text-4xl md:text-5xl font-black text-slate-950 tracking-tighter">
                      2. Context
                    </h1>
                    <div className="bg-slate-100 p-2 rounded-2xl flex gap-2 w-full md:w-auto">
                      <button
                        onClick={() => setDoshaMode("manual")}
                        className={`flex-1 md:flex-none px-6 py-3 rounded-xl font-bold transition-all ${doshaMode === "manual" ? "bg-white shadow-md text-emerald-700" : "text-slate-500 hover:bg-slate-200"}`}
                      >
                        Manual
                      </button>
                      <button
                        onClick={() => setDoshaMode("quiz")}
                        className={`flex-1 md:flex-none px-6 py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${doshaMode === "quiz" ? "bg-white shadow-md text-emerald-700" : "text-slate-500 hover:bg-slate-200"}`}
                      >
                        <BrainCircuit size={18} /> AI Calculator
                      </button>
                    </div>
                  </div>

                  {doshaMode === "manual" ? (
                    <div className="space-y-10 bg-slate-50 p-6 md:p-10 rounded-3xl border border-slate-200/50">
                      <div className="space-y-3">
                        <label className="text-lg font-bold text-slate-700 flex items-center gap-2">
                          <Wind size={20} className="text-emerald-500" /> Known
                          Dosha
                        </label>
                        <select
                          name="dosha"
                          value={formData.dosha}
                          onChange={handleInputChange}
                          className="w-full p-6 text-xl rounded-3xl border-2 border-slate-200/60 bg-white focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                        >
                          <option>Vata (Air/Space)</option>
                          <option>Pitta (Fire/Water)</option>
                          <option>Kapha (Water/Earth)</option>
                        </select>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-8 bg-emerald-50/50 p-6 md:p-10 rounded-3xl border-2 border-emerald-100">
                      <h3 className="text-2xl font-black text-emerald-900 mb-6 flex items-center gap-3">
                        <PieChart className="text-emerald-600" /> Algorithmic
                        Assessment
                      </h3>
                      <div className="space-y-3">
                        <label className="text-lg font-bold text-slate-700">
                          1. Body Frame
                        </label>
                        <select
                          value={quizAnswers.frame}
                          onChange={(e) =>
                            setQuizAnswers({
                              ...quizAnswers,
                              frame: e.target.value,
                            })
                          }
                          className="w-full p-6 text-xl rounded-3xl border-2 border-white bg-white focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                        >
                          <option value="">Select option...</option>
                          <option>Thin/Light</option>
                          <option>Medium/Athletic</option>
                          <option>Heavy/Solid</option>
                        </select>
                      </div>
                      <div className="space-y-3">
                        <label className="text-lg font-bold text-slate-700">
                          2. Digestion
                        </label>
                        <select
                          value={quizAnswers.digestion}
                          onChange={(e) =>
                            setQuizAnswers({
                              ...quizAnswers,
                              digestion: e.target.value,
                            })
                          }
                          className="w-full p-6 text-xl rounded-3xl border-2 border-white bg-white focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                        >
                          <option value="">Select option...</option>
                          <option>Irregular/Gas</option>
                          <option>Strong/Acidic</option>
                          <option>Slow/Sluggish</option>
                        </select>
                      </div>
                      <div className="space-y-3">
                        <label className="text-lg font-bold text-slate-700">
                          3. Sleep
                        </label>
                        <select
                          value={quizAnswers.sleep}
                          onChange={(e) =>
                            setQuizAnswers({
                              ...quizAnswers,
                              sleep: e.target.value,
                            })
                          }
                          className="w-full p-6 text-xl rounded-3xl border-2 border-white bg-white focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                        >
                          <option value="">Select option...</option>
                          <option>Light/Interrupted</option>
                          <option>Moderate/Sound</option>
                          <option>Deep/Prolonged</option>
                        </select>
                      </div>
                    </div>
                  )}

                  <div className="space-y-3 pt-6">
                    <label className="text-lg font-bold text-slate-700 flex items-center gap-2">
                      <Thermometer size={20} className="text-emerald-500" />{" "}
                      Current Season
                    </label>
                    <select
                      name="season"
                      value={formData.season}
                      onChange={handleInputChange}
                      className="w-full p-6 text-xl rounded-3xl border-2 border-slate-200/60 bg-slate-50 focus:ring-4 focus:ring-emerald-200 outline-none font-bold"
                    >
                      <option>Summer (Grishma)</option>
                      <option>Monsoon (Varsha)</option>
                      <option>Winter (Hemanta)</option>
                      <option>Spring (Vasanta)</option>
                    </select>
                  </div>
                </motion.div>
              )}

              {step === 3 && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="space-y-10"
                >
                  <h1 className="text-4xl md:text-5xl font-black text-slate-950 tracking-tighter">
                    3. Clinical Symptoms
                  </h1>
                  <textarea
                    name="symptoms"
                    value={formData.symptoms}
                    onChange={handleInputChange}
                    className="w-full p-8 h-80 text-xl md:text-2xl font-medium rounded-[2rem] border-2 border-slate-200/60 bg-slate-50 focus:ring-4 focus:ring-emerald-200 outline-none resize-none shadow-inner"
                    placeholder="e.g. persistent chills and high fever for 3 days..."
                  ></textarea>
                </motion.div>
              )}

              {error && (
                <div className="mt-8 p-6 bg-red-100/60 border border-red-200 text-red-700 font-bold rounded-3xl flex gap-3">
                  <AlertCircle size={22} className="flex-shrink-0" />
                  {error}
                </div>
              )}

              <div className="mt-16 flex justify-between items-center border-t-2 border-slate-100 pt-10">
                {step > 1 ? (
                  <button
                    onClick={() => setStep(step - 1)}
                    className="px-6 py-4 font-black text-slate-400 hover:text-slate-900 text-lg"
                  >
                    Back
                  </button>
                ) : (
                  <div />
                )}
                {step < 3 ? (
                  <button
                    onClick={handleNext}
                    className="bg-slate-950 text-white px-8 py-4 rounded-3xl font-black text-xl flex gap-3 hover:bg-slate-800 shadow-xl"
                  >
                    Next <ChevronRight size={22} />
                  </button>
                ) : (
                  <button
                    onClick={runAnalysis}
                    disabled={!formData.symptoms.trim()}
                    className="bg-gradient-to-r from-emerald-600 to-teal-600 text-white px-8 py-4 rounded-3xl font-black text-lg md:text-2xl flex gap-3 shadow-xl disabled:opacity-50"
                  >
                    Run Diagnostic <Sparkles size={24} />
                  </button>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* LOADING SCREEN */}
      {isAnalyzing && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 min-h-screen bg-slate-950/80 backdrop-blur-xl flex items-center justify-center z-[300] p-6"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center p-12 md:p-20 space-y-8 bg-white/60 backdrop-blur-xl rounded-[4rem] shadow-2xl border-4 border-white text-center"
          >
            <div className="relative w-32 h-32 md:w-40 md:h-40 flex items-center justify-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                className="absolute inset-0 border-8 border-emerald-100 border-t-emerald-600 rounded-full"
              />
              <BrainCircuit className="text-emerald-600 w-12 h-12 md:w-16 md:h-16 animate-pulse" />
            </div>
            <h3 className="text-3xl md:text-4xl font-black text-slate-950 tracking-tighter">
              Synthesizing Hybrid Intelligence
            </h3>
          </motion.div>
        </motion.div>
      )}

      {/* RESULTS DASHBOARD */}
      {showResult && result && (
        <motion.div
          key="result"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-12 pb-20"
        >
          <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 no-print">
            <div>
              <button
                onClick={() => setShowResult(false)}
                className="text-emerald-600 font-black text-sm mb-4 hover:underline flex gap-1"
              >
                - Start New Analysis
              </button>
              <h2 className="text-4xl md:text-5xl font-black tracking-tighter text-slate-950">
                Report: {formData.name}
              </h2>
            </div>
            <div className="flex flex-col sm:flex-row gap-4 w-full md:w-auto">
              <button
                onClick={() => window.print()}
                className="bg-white border-2 border-slate-200/60 shadow-lg px-8 py-4 rounded-full font-black text-lg flex items-center justify-center gap-3 hover:bg-slate-50 w-full sm:w-auto"
              >
                <Download size={22} /> Export PDF
              </button>
              <button
                onClick={saveToCloud}
                disabled={savedSuccess || isSaving}
                className={`px-8 py-4 rounded-full font-black text-lg flex items-center justify-center gap-3 shadow-lg transition-all w-full sm:w-auto ${savedSuccess ? "bg-teal-600 text-white" : "bg-slate-950 text-white hover:bg-slate-800"}`}
              >
                {isSaving ? (
                  "Saving..."
                ) : savedSuccess ? (
                  <>
                    <CheckCircle2 size={22} /> Saved to Cloud
                  </>
                ) : (
                  <>
                    <Save size={22} /> Save Record
                  </>
                )}
              </button>
            </div>
          </div>

          {isLowConfidence && (
            <div className="bg-orange-50 border-2 border-orange-200 p-6 rounded-2xl flex gap-4 items-center mb-4">
              <AlertCircle className="text-orange-600 w-8 h-8 flex-shrink-0" />
              <div>
                <h4 className="font-black text-orange-900 text-lg">
                  Clinical Review Required
                </h4>
                <p className="text-orange-800 font-medium">
                  The patient's symptoms are too broad. The AI confidence is
                  below the safety threshold. Do not prescribe solely based on
                  this prediction.
                </p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
            <div className="lg:col-span-1 space-y-10">
              <div
                className={`text-white p-10 rounded-[3rem] shadow-2xl relative overflow-hidden print-shadow-none ${isLowConfidence ? "bg-orange-950" : "bg-slate-950"}`}
              >
                <div className="absolute -right-4 -bottom-4 opacity-10">
                  <Activity size={200} />
                </div>
                <p className="text-slate-400 text-sm font-black uppercase tracking-widest mb-3">
                  ML Decision
                </p>
                <h3
                  className={`text-4xl md:text-5xl font-black mb-10 leading-tight ${isLowConfidence ? "text-orange-400" : "text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-300"}`}
                >
                  {isLowConfidence ? "Inconclusive Data" : result.prediction}
                </h3>
                <div className="flex items-center justify-between mt-10 border-t border-slate-700/50 pt-10">
                  <div>
                    <p className="text-slate-400 text-lg font-bold">
                      Confidence
                    </p>
                    <p
                      className={`text-4xl md:text-5xl font-black ${isLowConfidence ? "text-orange-500" : ""}`}
                    >
                      {result.confidence}%
                    </p>
                  </div>
                </div>
              </div>

              {/* AI X-RAY CARD */}
              <div className="bg-white p-10 rounded-[3rem] border border-slate-200/60 shadow-sm print-shadow-none">
                <h4 className="font-black text-slate-950 mb-8 text-xl flex items-center gap-2">
                  <ActivitySquare className="text-emerald-500" /> AI X-Ray
                </h4>
                <div className="space-y-6">
                  {result.xai_breakdown &&
                    result.xai_breakdown.map((feature, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between text-sm font-bold text-slate-700 mb-2">
                          <span className="uppercase">{feature.symptom}</span>
                          <span className="text-emerald-600">
                            {feature.weight}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-3">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${feature.weight}%` }}
                            transition={{ duration: 1, delay: idx * 0.2 }}
                            className="bg-gradient-to-r from-emerald-400 to-teal-500 h-3 rounded-full"
                          />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>

            <div className="lg:col-span-2 space-y-10">
              <div className="bg-white p-8 md:p-12 rounded-[3rem] border border-slate-200/60 shadow-sm flex flex-col md:flex-row justify-between print-shadow-none gap-6">
                <div>
                  <p className="text-slate-500 font-bold uppercase tracking-widest text-sm mb-2">
                    Calculated Dosha Matrix
                  </p>
                  <p className="text-2xl font-black text-slate-900">
                    {formData.dosha}
                  </p>
                </div>
                <div className="md:text-right">
                  <p className="text-slate-500 font-bold uppercase tracking-widest text-sm mb-2">
                    Age / Sex
                  </p>
                  <p className="text-2xl font-black text-slate-900">
                    {formData.age} yrs / {formData.gender}
                  </p>
                </div>
              </div>

              {/* CONTEXTUAL REASONING CARD */}
              <div className="bg-white p-8 md:p-16 rounded-[3rem] border border-slate-200/60 shadow-sm print-shadow-none">
                <h4 className="font-black text-slate-950 mb-8 md:mb-10 flex items-center gap-3 text-2xl md:text-3xl">
                  <BrainCircuit className="text-emerald-600" /> Ayurvedic AI Contextual Analysis
                </h4>
                
                {result.recommendation ? (
                  <div className="prose max-w-none text-slate-800 leading-relaxed text-lg whitespace-pre-wrap font-semibold">
                    {result.recommendation.replace(/\*\*/g, '')}
                  </div>
                ) : (
                  <>
                    <p className="text-slate-800 leading-relaxed text-lg md:text-xl font-semibold mb-12">
                      {result.reasoning}
                    </p>
                    <div className="my-12 h-0.5 bg-slate-100" />
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                      <div>
                        <h5 className="font-black text-slate-950 flex gap-2 text-xl md:text-2xl mb-8">
                          <Leaf className="text-emerald-500" /> Protocol Herbs
                        </h5>
                        <ul className="space-y-6">
                          {result.herbs &&
                            result.herbs.map((herb: any, idx: number) => (
                              <li
                                key={idx}
                                className="flex gap-4 items-start bg-slate-50 p-5 rounded-2xl"
                              >
                                <div className="mt-1 bg-emerald-100 p-1.5 rounded-full flex-shrink-0">
                                  <CheckCircle2
                                    size={18}
                                    className="text-emerald-600"
                                  />
                                </div>
                                <div>
                                  <strong className="block text-slate-900 text-lg font-black">
                                    {herb.name}
                                  </strong>
                                  <span className="text-slate-600 text-sm font-bold">
                                    {herb.benefit}
                                  </span>
                                </div>
                              </li>
                            ))}
                        </ul>
                      </div>
                      <div>
                        <h5 className="font-black text-slate-950 flex gap-2 text-xl md:text-2xl mb-8">
                          <ClipboardCheck className="text-emerald-500" /> Lifestyle
                          Support
                        </h5>
                        <ul className="space-y-6">
                          {result.lifestyle &&
                            result.lifestyle.map((item: string, idx: number) => (
                              <li
                                key={idx}
                                className="flex gap-4 items-start bg-slate-50 p-5 rounded-2xl"
                              >
                                <div className="mt-1 bg-emerald-100 p-1.5 rounded-full flex-shrink-0">
                                  <CheckCircle2
                                    size={18}
                                    className="text-emerald-600"
                                  />
                                </div>
                                <span className="text-slate-800 text-sm font-bold">
                                  {item}
                                </span>
                              </li>
                            ))}
                        </ul>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

// --- MAIN APP ROUTER ---
export default function App() {
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [userData, setUserData] = useState<UserData | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [selectedRole, setSelectedRole] = useState<"patient" | "doctor">(
    "patient",
  );
  const [clinicIdInput, setClinicIdInput] = useState("");

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        const userDocRef = doc(db, "users", currentUser.uid);
        const userDocSnap = await getDoc(userDocRef);

        if (userDocSnap.exists()) {
          setUserData(userDocSnap.data() as UserData);
        }
        setUser(currentUser);
      } else {
        setUser(null);
        setUserData(null);
      }
      setAuthLoading(false);
    });
    return () => unsubscribe();
  }, []);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError("");
    try {
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        if (selectedRole === "patient" && clinicIdInput.trim().length !== 6) {
          setAuthError(
            "Patients must enter a valid 6-character Clinic ID provided by their doctor.",
          );
          return;
        }

        const userCredential = await createUserWithEmailAndPassword(
          auth,
          email,
          password,
        );

        let assignedClinicId =
          selectedRole === "doctor"
            ? Math.random().toString(36).substring(2, 8).toUpperCase()
            : clinicIdInput.trim().toUpperCase();

        await setDoc(doc(db, "users", userCredential.user.uid), {
          email: userCredential.user.email,
          role: selectedRole,
          clinicId: assignedClinicId,
        });
      }
    } catch (err: any) {
      setAuthError(err.message.replace("Firebase: ", ""));
    }
  };

  const handleGoogleAuth = async () => {
    setAuthError("");
    const provider = new GoogleAuthProvider();
    try {
      if (
        !isLogin &&
        selectedRole === "patient" &&
        clinicIdInput.trim().length !== 6
      ) {
        setAuthError(
          "Patients must enter a valid 6-character Clinic ID provided by their doctor before continuing with Google.",
        );
        return;
      }

      const userCredential = await signInWithPopup(auth, provider);
      const userDocRef = doc(db, "users", userCredential.user.uid);
      const userDocSnap = await getDoc(userDocRef);

      if (!userDocSnap.exists()) {
        let assignedClinicId =
          selectedRole === "doctor"
            ? Math.random().toString(36).substring(2, 8).toUpperCase()
            : clinicIdInput.trim().toUpperCase();

        await setDoc(userDocRef, {
          email: userCredential.user.email,
          role: selectedRole,
          clinicId: assignedClinicId,
        });
      }
    } catch (err: any) {
      setAuthError(err.message.replace("Firebase: ", ""));
    }
  };

  if (authLoading)
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center text-emerald-500 font-black text-2xl">
        Initializing Secure Portal...
      </div>
    );

  if (!user) {
    return (
      <div className="min-h-screen bg-slate-900 grid grid-cols-1 lg:grid-cols-2">
        <div className="bg-emerald-950 p-10 lg:p-20 flex flex-col justify-between relative overflow-hidden hidden lg:flex">
          <div className="absolute inset-0 opacity-10 bg-[url('https://grainy-gradients.vercel.app/gradients/6.png')] bg-cover"></div>
          <div className="relative z-10 flex items-center gap-3">
            <div className="bg-gradient-to-br from-emerald-500 to-teal-700 p-3 rounded-2xl">
              <Leaf className="text-white w-8 h-8" />
            </div>
            <span className="text-3xl font-black tracking-tight text-white">
              Arogya<span className="text-emerald-400">AI</span>
            </span>
          </div>
          <div className="relative z-10 space-y-6">
            <h1 className="text-5xl lg:text-6xl font-black tracking-tighter text-white leading-tight">
              Hybrid Intelligence for Ayurveda
            </h1>
            <p className="text-emerald-200/80 font-medium text-lg lg:text-xl max-w-xl">
              A professional assistant combining traditional Machine Learning
              accuracy with Generative AI reasoning.
            </p>
          </div>
          <div className="relative z-10 text-emerald-600 text-sm font-medium">
            {new Date().getFullYear()} Secure Cloud Dashboard.
          </div>
        </div>

        <div className="bg-slate-50 p-6 sm:p-10 lg:p-20 flex flex-col items-center justify-center">
          <div className="max-w-md w-full space-y-8">
            <div className="space-y-3 text-center lg:text-left">
              <h2 className="text-4xl lg:text-5xl font-black text-slate-950 tracking-tighter">
                {isLogin ? "Welcome Back" : "Create Account"}
              </h2>
              <p className="text-slate-500 text-base lg:text-lg font-medium">
                {isLogin
                  ? "Sign in to access your portal."
                  : "Join the ArogyaAI network."}
              </p>
            </div>

            {!isLogin && (
              <div className="bg-slate-200/50 p-1.5 rounded-2xl flex relative shadow-inner">
                <div
                  className={`absolute top-1.5 bottom-1.5 w-[calc(50%-6px)] bg-white rounded-xl shadow-sm transition-transform duration-300 ease-in-out ${selectedRole === "doctor" ? "translate-x-full left-0" : "translate-x-0 left-1.5"}`}
                ></div>
                <button
                  onClick={() => setSelectedRole("patient")}
                  className={`flex-1 py-3 font-bold text-sm z-10 transition-colors ${selectedRole === "patient" ? "text-emerald-600" : "text-slate-500"}`}
                >
                  I am a Patient
                </button>
                <button
                  onClick={() => setSelectedRole("doctor")}
                  className={`flex-1 py-3 font-bold text-sm z-10 transition-colors ${selectedRole === "doctor" ? "text-emerald-600" : "text-slate-500"}`}
                >
                  I am a Practitioner
                </button>
              </div>
            )}

            {authError && (
              <div className="p-4 bg-red-100 text-red-700 font-bold rounded-xl text-sm">
                {authError}
              </div>
            )}

            <form onSubmit={handleAuth} className="space-y-4">
              <div className="relative">
                <Mail className="absolute left-5 top-4 text-slate-400 w-6 h-6" />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full p-4 pl-14 rounded-2xl border border-slate-200/60 bg-white focus:ring-4 focus:ring-emerald-200 outline-none font-bold shadow-inner"
                  placeholder="Email Address"
                />
              </div>
              <div className="relative">
                <Lock className="absolute left-5 top-4 text-slate-400 w-6 h-6" />
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full p-4 pl-14 rounded-2xl border border-slate-200/60 bg-white focus:ring-4 focus:ring-emerald-200 outline-none font-bold shadow-inner"
                  placeholder="Password (min 6 chars)"
                />
              </div>

              {!isLogin && selectedRole === "patient" && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="relative"
                >
                  <Building className="absolute left-5 top-4 text-slate-400 w-6 h-6" />
                  <input
                    type="text"
                    required
                    value={clinicIdInput}
                    onChange={(e) => setClinicIdInput(e.target.value)}
                    maxLength={6}
                    className="w-full p-4 pl-14 rounded-2xl border-2 border-emerald-200/60 bg-emerald-50 focus:ring-4 focus:ring-emerald-200 outline-none font-black text-emerald-900 tracking-widest uppercase shadow-inner"
                    placeholder="Enter 6-Digit Clinic ID"
                  />
                  <p className="text-xs font-bold text-slate-400 mt-2 ml-2">
                    Ask your doctor for their specific Clinic ID code to link
                    your accounts.
                  </p>
                </motion.div>
              )}

              <button
                type="submit"
                className="w-full bg-slate-950 text-white py-4 rounded-2xl font-black text-xl flex items-center justify-center gap-3 hover:bg-slate-800 transition-all shadow-xl shadow-slate-950/20 mt-4"
              >
                {isLogin ? "Sign In" : "Register"} <LogIn size={22} />
              </button>
            </form>

            <div className="relative flex items-center py-2">
              <div className="flex-grow border-t border-slate-200"></div>
              <span className="flex-shrink-0 mx-4 text-slate-400 font-bold text-xs uppercase tracking-widest">
                OR
              </span>
              <div className="flex-grow border-t border-slate-200"></div>
            </div>

            <button
              type="button"
              onClick={handleGoogleAuth}
              className="w-full bg-white text-slate-700 border-2 border-slate-200 py-4 rounded-2xl font-black text-lg flex items-center justify-center gap-4 hover:bg-slate-50 transition-all shadow-sm"
            >
              <svg className="w-6 h-6" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              Continue with Google
            </button>

            <button
              onClick={() => setIsLogin(!isLogin)}
              className="w-full text-center text-slate-500 font-bold text-sm hover:text-emerald-600 transition-colors"
            >
              {isLogin
                ? "Don't have an account? Register here."
                : "Already have an account? Sign in."}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // --- MAIN APP ROUTING LOGIC BASED ON ROLE ---
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-slate-50/50 font-sans text-slate-900 selection:bg-emerald-100 overflow-x-hidden">
        <AnimatePresence>
          {isSidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsSidebarOpen(false)}
              className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-[90] md:hidden"
            />
          )}
        </AnimatePresence>

        <Sidebar
          user={user}
          userData={userData}
          isOpen={isSidebarOpen}
          setIsOpen={setIsSidebarOpen}
        />

        <main
          className={`flex-1 transition-all duration-300 ease-in-out ${isSidebarOpen ? "md:ml-72" : "ml-0"}`}
        >
          <div className="p-4 md:p-6 lg:px-10 lg:pt-10 flex items-center no-print">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="bg-white border-2 border-slate-200 p-2 md:p-2.5 rounded-xl text-slate-600 hover:border-emerald-500 hover:text-emerald-600 transition-all shadow-sm"
            >
              <Menu size={20} />
            </button>
          </div>

          <div className="-mt-4">
            <Routes>
              {userData?.role === "doctor" ? (
                <>
                  <Route path="/" element={<GlobalDashboard />} />
                  <Route
                    path="/diagnose"
                    element={<DiagnosticTool userData={userData} />}
                  />
                  <Route
                    path="/patients"
                    element={<PatientRecords userData={userData} />}
                  />
                </>
              ) : (
                <>
                  <Route
                    path="/"
                    element={
                      <PatientDashboard user={user} userData={userData} />
                    }
                  />
                  <Route
                    path="/checkup"
                    element={<PatientCheckup user={user} userData={userData} />}
                  />
                  <Route
                    path="/my-records"
                    element={
                      <div className="p-10">
                        <h1 className="text-3xl font-black">
                          My Personal Records
                        </h1>
                        <p className="text-slate-500">Feature coming soon.</p>
                      </div>
                    }
                  />
                </>
              )}

              <Route
                path="/profile"
                element={<ProfileSettings user={user} userData={userData} />}
              />
              <Route path="/help" element={<HelpCenter />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}
