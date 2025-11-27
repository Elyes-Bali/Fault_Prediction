// DataChart.jsx
import React, { useState, useCallback, useEffect } from 'react';
import { 
    LineChart, Line, XAxis, YAxis, CartesianGrid, 
    Tooltip, Legend, ResponsiveContainer, Brush 
} from 'recharts';

// Define a color palette for the lines
const COLORS = [
    "#8884d8", // Blue-Purple
    "#82ca9d", // Green
    "#ffc658", // Yellow-Orange
    "#ff7300", // Dark Orange
    "#0088FE", // Bright Blue
    "#00C49F", // Teal
    "#FFBB28", // Gold
    "#FF8042", // Salmon
    "#A100F2", // Violet
    "#2C3E50"  // Dark Grey (for more features)
];


const DataChart = ({ jsonString, title }) => {
    if (!jsonString) return null;

    let data;
    try {
        data = JSON.parse(jsonString);
    } catch (e) {
        return <p style={{ color: 'red' }}>Error parsing chart data: Invalid JSON format.</p>;
    }

    const { columns, data: rows } = data;

    if (!columns || !rows || rows.length === 0) {
        return <p>No data available for charting.</p>;
    }

    // --- Data Sampling ---
    const MAX_POINTS = 1000; 
    const sampledRows = rows.slice(0, MAX_POINTS);
    
    // Identify the datetime column
    const datetimeKey = 'datetime';
    const datetimeIndex = columns.indexOf(datetimeKey);
    
    // --- Feature Filtering ---
    const featureColumns = columns.filter(col => 
        col !== datetimeKey && 
        col !== 'index' &&
        col !== 'unnamed' && 
        col !== 'Unnamed: 0' && 
        col !== 'Unnamed: 0.1' && 
        col !== 'id'
    );

    if (datetimeIndex === -1 || featureColumns.length === 0) {
        return <p>Missing required 'datetime' column or any feature columns for charting.</p>;
    }
    
    // --- State for Line Toggling ---
    const [visibleLines, setVisibleLines] = useState(() => 
        featureColumns.reduce((acc, col) => {
            acc[col] = true; 
            return acc;
        }, {})
    );

    // Reset visibility state when new data is loaded
    useEffect(() => {
        setVisibleLines(
            featureColumns.reduce((acc, col) => {
                acc[col] = true;
                return acc;
            }, {})
        );
    }, [jsonString]);
    
    const handleLegendClick = useCallback((e) => {
        const dataKey = e.dataKey;
        setVisibleLines(prev => ({
            ...prev,
            [dataKey]: !prev[dataKey] 
        }));
    }, []);
    // --- End State for Line Toggling ---

    // Convert pandas split format to recharts array-of-objects format
    const chartData = sampledRows.map(row => {
        const obj = {};
        obj[datetimeKey] = row[datetimeIndex];
        
        featureColumns.forEach(featureName => {
            const featureIndex = columns.indexOf(featureName);
            const featureValue = parseFloat(row[featureIndex]);
            obj[featureName] = isNaN(featureValue) ? null : featureValue;
        });

        return obj;
    });
    
    // Dynamic Line component generation: create a Line for every feature column
    const lines = featureColumns.map((key, index) => (
        <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={COLORS[index % COLORS.length]}
            dot={false}
            strokeWidth={1}
            isAnimationActive={false}
            hide={!visibleLines[key]} 
        />
    ));


    return (
      <div className="my-8 p-6 bg-white shadow-xl rounded-xl border border-gray-200">
            {/* FIX 2: Updated text colors for light background */}
            <h3 className="text-xl font-medium text-gray-800 mb-2">{title}</h3> 
            <p className="text-sm text-gray-600 mb-4">
                Chart displaying **{featureColumns.length} feature(s)** vs. Time. 
                (Showing **{sampledRows.length}** of **{rows.length}** points for clarity. Click legend to toggle lines.)
            </p>
            
            <ResponsiveContainer width="100%" height={400}>
                <LineChart
                    data={chartData}
                    margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
                >
                    {/* FIX 3: Changed grid stroke to a very light gray */}
                    <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" /> 
                    
                    <XAxis 
                        dataKey={datetimeKey} 
                        angle={-15} 
                        textAnchor="end" 
                        height={60}
                        // FIX 4: Changed axis stroke to dark gray
                        stroke="#374151"
                        tickFormatter={(tick) => {
                            const date = new Date(tick);
                            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }); 
                        }}
                    />
                    
                    <YAxis 
                        // FIX 5: Changed label fill to dark gray
                        label={{ value: "Sensor Value", angle: -90, position: 'insideLeft', fill: '#374151' }} 
                        type="number"
                        domain={[-7, 250]} 
                        // FIX 6: Changed axis stroke to dark gray
                        stroke="#374151"
                    />
                    
                    <Tooltip 
                        labelFormatter={(label) => new Date(label).toLocaleString()}
                        // FIX 7: Updated tooltip style for light background
                        contentStyle={{ backgroundColor: '#fff', border: '1px solid #ccc', color: '#1F2937' }}
                    />
                    
                    <Legend onClick={handleLegendClick} />
                    
                    <Brush 
                        dataKey={datetimeKey} 
                        height={30}
                        stroke="#8884d8"
                        // FIX 8: Changed brush fill to light gray
                        fill="#F3F4F6"
                        interval={5} 
                    />
                    
                    {lines}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default DataChart;