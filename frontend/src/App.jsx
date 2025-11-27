import { useState } from "react";
import { Route, Routes } from "react-router-dom";
import Classification from "./components/Classification";
import Regression from "./components/Regression";

function App() {
  return (
    <div>
      <Routes>
        <Route path="/" element={<Classification />} />
        <Route path="/reg" element={<Regression />} />
      </Routes>
    </div>
  );
}

export default App;
