// Regression.jsx
import React, { useState, useCallback } from "react";

import { motion } from "framer-motion";
import {
  Loader2,
  Upload,
  PlayCircle,
  BarChart3,
  FileText,
  Terminal,
  Settings,
  Info,
} from "lucide-react";
import Sidebar from "./Sidebar";
import TableDisplay from "./regresion/TableDisplay";
import DataChart from "./regresion/DataChart";

// IMPORTANT: Ensure this matches your Python API file
const API_URL = "http://127.0.0.1:4000/api";

const Regression = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [initialData, setInitialData] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [detailedData, setDetailedData] = useState(null);
  const [logs, setLogs] = useState("Awaiting CSV file upload...");
  const [isRunning, setIsRunning] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 },
  };

  const handleFileChange = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setInitialData(null);
    setSummaryData(null);
    setDetailedData(null);
    setErrorMessage("");
    setLogs(`📁 File selected: ${file.name}\nUploading...`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/upload-preview`, {
        method: "POST",
        body: formData,
      });
      const result = await response.json();

      if (response.ok && result.status === "preview_ready") {
        setInitialData(result.preview_data);
        setLogs((prev) => prev + "\n✔ Preview loaded successfully.");
      } else {
        setErrorMessage(result.error || "Unknown preview error.");
      }
    } catch {
      setErrorMessage("Network error during preview.");
    }
  }, []);

  const handleRunModel = useCallback(async () => {
    if (!selectedFile) return setErrorMessage("Please select a file first.");

    setIsRunning(true);
    setErrorMessage("");
    setSummaryData(null);
    setDetailedData(null);
    setLogs((prev) => prev + "\n🚀 Starting GRU prediction...");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_URL}/run-model`, {
        method: "POST",
        body: formData,
      });
      const result = await response.json();

      setLogs(result.logs || "No logs received.");
      if (result.status === "success") {
        setSummaryData(result.summary_data);
        setDetailedData(result.detailed_data);
      } else {
        setErrorMessage(result.error_message || "Inference failed.");
      }
    } catch {
      setErrorMessage("Network error during model run.");
    } finally {
      setIsRunning(false);
    }
  }, [selectedFile]);

  return (
    <div className="relative flex min-h-screen w-full">
      <Sidebar />
      {/* 🌌 Background (Kept the stylish background) */}
      <div className="absolute inset-0 -z-10 pointer-events-none overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-indigo-50 via-white to-purple-50"></div>
        <motion.div
          className="absolute w-[750px] h-[750px] rounded-full blur-3xl opacity-30 bg-indigo-500"
          animate={{ y: [0, -60, 0], x: [0, 60, 0], scale: [1, 1.15, 1] }}
          transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
          style={{ top: "-200px", left: "-150px" }}
        />
        <motion.div
          className="absolute w-[600px] h-[600px] rounded-full blur-3xl opacity-25 bg-violet-500"
          animate={{ y: [0, 50, 0], x: [0, -50, 0], scale: [1, 1.1, 1] }}
          transition={{ duration: 14, repeat: Infinity, ease: "easeInOut" }}
          style={{ bottom: "-150px", right: "-100px" }}
        />
        {[...Array(15)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 rounded-full bg-indigo-400 opacity-30"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
            animate={{ y: [0, -100, 0], opacity: [0.3, 0.6, 0.3] }}
            transition={{
              duration: 8 + Math.random() * 5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>

      {/* 💡 Main Content */}
      <motion.div
        className="flex-1 max-w-7xl mx-auto px-8 py-10"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl font-semibold text-center mb-10 text-gray-800 tracking-tight">
          📈 GRU Regression Dashboard
        </h1>

        {/* Step 1 - File Upload */}
        <motion.div
          className="mb-8 bg-white/80 shadow-lg backdrop-blur-md border border-gray-200 rounded-xl p-6 hover:shadow-xl transition"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
        >
          <h2 className="text-xl font-medium text-gray-700 mb-4">
            📥 1. Upload Time Series Dataset
          </h2>

          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            <label className="flex items-center gap-2 border rounded-lg px-4 py-2 bg-gray-100 hover:bg-gray-200 cursor-pointer">
              <Upload className="w-5" />
              <span>{selectedFile ? selectedFile.name : "Select CSV File"}</span>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            <motion.button
              onClick={handleRunModel}
              disabled={!selectedFile || isRunning}
              whileHover={{ scale: !isRunning ? 1.03 : 1 }}
              whileTap={{ scale: !isRunning ? 0.97 : 1 }}
              className={`flex items-center gap-2 px-6 py-2 rounded-lg text-white font-medium transition ${
                isRunning || !selectedFile
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-indigo-600 hover:bg-indigo-700"
              }`}
            >
              {isRunning ? (
                <Loader2 className="animate-spin" />
              ) : (
                <PlayCircle />
              )}
              {isRunning ? "Running..." : "Run GRU Model"}
            </motion.button>
          </div>

          {errorMessage && (
            <p className="mt-3 text-red-500 text-sm">⚠ {errorMessage}</p>
          )}
        </motion.div>

        {/* File Preview & Chart */}
        {initialData && (
          <>
            <TableDisplay
              jsonString={initialData}
              title="📊 Uploaded CSV Preview"
            />
            <motion.div variants={itemVariants}>
              <DataChart
                jsonString={initialData}
                title="Uploaded CSV Data Chart (All Features)"
              />
            </motion.div>
          </>
        )}

        {/* Logs */}
        <motion.div className="mt-8">
          <h2 className="text-xl font-medium text-gray-700 mb-3">
            📟 2. Live Logs
          </h2>
          <div className="bg-black rounded-lg p-4 text-green-400 font-mono h-64 overflow-y-auto relative shadow-inner">
            <pre>{logs}</pre>
            <span className="absolute bottom-2 left-2 animate-pulse">▋</span>
          </div>
        </motion.div>

        {/* Results */}
        <motion.div
          className="mt-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {(summaryData || detailedData) && (
            <h2 className="text-xl font-medium text-gray-700 mb-4">
              📊 3. Regression Results
            </h2>
          )}

          <TableDisplay
            jsonString={summaryData}
            title="📉 Model Performance Metrics"
            // Applying column exclusion logic to the summary table
            columnsToExclude={["Duration_Windows", "Duration"]} 
          />
          
          <DataChart
            jsonString={detailedData}
            title="🔍 Actual vs. Predicted Values"
          />
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Regression;