// src/components/Sidebar.jsx
import React from "react";
import {
  X,
  Home,
  Users,
  Settings,
  Briefcase,
  ChevronRight,
  GraduationCap,
  Gem,
  Info,
  Upload,
  Terminal,
  BarChart3
} from "lucide-react";

import { motion } from "framer-motion";
import { href } from "react-router-dom";


const navigation = [
  { name: "Classification", icon: Home,href:"/"  },
  { name: "Regression", icon: Upload,href:"/reg" },
  { name: "Logs", icon: Terminal },
  { name: "Results", icon: BarChart3 },

];

const Sidebar = ({ isOpen, toggleSidebar }) => {

  return (
    <>
      {/* Overlay for mobile */}
      <div
        className={`fixed inset-0 bg-gray-900/50 z-30 transition-opacity duration-300 lg:hidden 
        ${
          isOpen
            ? "opacity-100 pointer-events-auto"
            : "opacity-0 pointer-events-none"
        }`}
        onClick={toggleSidebar}
      ></div>

      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 w-64 bg-gray-800 text-white shadow-2xl z-40
        transform transition-transform duration-300 ease-in-out
        lg:translate-x-0 ${isOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        <div className="flex items-center  p-4 border-b border-gray-700/50 h-16">
          <GraduationCap className="w-8 h-8 mr-2 text-orange-500 " />

          <div className="text-xl font-extrabold tracking-wider text-indigo-100">
            Dashboard
          </div>
          <button
            onClick={toggleSidebar}
            className="p-2 lg:hidden text-gray-400 hover:text-white rounded-full hover:bg-gray-700 transition"
          >
            <X size={20} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          {navigation.map((item) => (
            <a
              key={item.name}
              href={item.href}
              className={`flex items-center p-3 rounded-xl transition-all duration-200 group
                ${
                  item.name === "Dashboard"
                    ? "bg-indigo-600 shadow-lg text-white font-semibold"
                    : "text-gray-300 hover:bg-gray-700 hover:text-white"
                }`}
            >
              <item.icon className="h-5 w-5 mr-3" />
              <span>{item.name}</span>
              <ChevronRight
                className={`ml-auto h-4 w-4 transition-transform duration-200 
                ${
                  item.name === "Dashboard"
                    ? "opacity-100"
                    : "opacity-0 group-hover:opacity-100"
                }`}
              />
            </a>
          ))}
        </nav>

        {/* Footer */}
        {/* <div className="absolute bottom-0 left-0 w-full p-4 border-t border-gray-700/50">
          <div className="flex items-center p-3 rounded-lg bg-gray-700/50">
            <div className="h-8 w-8 rounded-full overflow-hidden mr-3">
              <img
                src={
                  user?.profileImage ||
                  "https://placehold.co/40x40/4F46E5/ffffff?text=User"
                }
                alt={user?.name || "User Avatar"}
                className="h-full w-full object-cover"
              />
            </div>

            <div>
              <p className="text-sm font-medium text-white">{user.name}</p>
              <p className="text-xs text-gray-400">{user.email}</p>
            </div>
          </div>

          <div className="pt-4 border-t border-gray-800">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={logout}
              className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-red-600 text-white font-bold rounded-lg shadow-lg hover:bg-red-700 transition"
            >
              <FiLogOut className="w-5 h-5" />
              <span>Logout</span>
            </motion.button>
          </div>
        </div> */}
      </div>
    </>
  );
};

export default Sidebar;
