import React, { useState, useEffect } from 'react'; 
import { motion } from 'framer-motion';

const TableDisplay = ({ jsonString, title,columnsToExclude = [] }) => {
    if (!jsonString) return null;

    let data;
    try {
        data = JSON.parse(jsonString);
    } catch (e) {
        return <p className="text-red-500">Error parsing data: Invalid JSON format.</p>;
    }

    const { columns, data: allRows } = data;
    const filteredColumns = columns.filter(col => !columnsToExclude.includes(col));
    
    // 2. Map original column indices to the filtered columns
    const columnIndicesToKeep = filteredColumns.map(col => columns.indexOf(col));
    
    // 3. Filter the data rows to only include cells from the kept columns
    const filteredRows = allRows.map(row => 
        columnIndicesToKeep.map(index => row[index])
    );

    // Reset page state whenever new data loads
    const [currentPage, setCurrentPage] = useState(1);
    
    // useEffect(() => {
    //     setCurrentPage(1);
    // }, [jsonString]); // Reset to page 1 whenever the input data changes

    // if (!columns || !allRows || allRows.length === 0) {
    //     return <p className="text-gray-500 dark:text-gray-400 my-4 p-4 border rounded-lg bg-white dark:bg-gray-800">{title}: No data to display.</p>;
    // }

  useEffect(() => {
        setCurrentPage(1);
    }, [jsonString]); // Reset to page 1 whenever the input data changes

    // Use filteredRows.length to check if there is data to display
    if (!filteredColumns || filteredRows.length === 0) { 
        // Only check allRows if columns are still present, otherwise check filteredRows
        if (!columns || allRows.length === 0) {
             return <p className="text-gray-500 dark:text-gray-400 my-4 p-4 border rounded-lg bg-white dark:bg-gray-800">{title}: No data to display.</p>;
        }
    }

    // --- PAGINATION LOGIC ---
    const rowsPerPage = 10;
    const totalPages = Math.ceil(filteredRows.length / rowsPerPage); // Use filteredRows length

    // Calculate start and end index for the current page
    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;
    const rowsToDisplay = filteredRows.slice(startIndex, endIndex); // Use filteredRows
    // --- END PAGINATION LOGIC ---

    const handleNextPage = () => {
        if (currentPage < totalPages) {
            setCurrentPage(currentPage + 1);
        }
    };

    const handlePrevPage = () => {
        if (currentPage > 1) {
            setCurrentPage(currentPage - 1);
        }
    };
    // --- END PAGINATION LOGIC ---
    
    // Framer Motion Variants (NO CHANGE)
    const tableVariants = {
        hidden: { opacity: 0, y: 50 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } }
    };

    return (
    <motion.div 
            className="my-8 p-4 bg-white dark:bg-gray-800 shadow-xl rounded-xl border border-gray-200 dark:border-gray-700"
            variants={tableVariants}
            initial="hidden"
            animate="visible"
        >
            <h3 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-300 border-b pb-2">
                {title} <span className="text-sm font-normal text-indigo-500">({allRows.length} total rows)</span>
            </h3>

            {/* Pagination Controls (NO CHANGE) */}
            {filteredRows.length > rowsPerPage && ( // Use filteredRows length for pagination check
                <div className="flex justify-center items-center mb-4 space-x-2">
                    <button
                        onClick={handlePrevPage}
                        disabled={currentPage === 1}
                        className="px-3 py-1 text-sm bg-indigo-500 text-white rounded-md hover:bg-indigo-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                    >
                        Previous
                    </button>
                    
                    <span className="text-gray-600 dark:text-gray-300 text-sm font-medium">
                        Page {currentPage} of {totalPages} (Rows {startIndex + 1} - {Math.min(endIndex, filteredRows.length)})
                    </span>
                    
                    <button
                        onClick={handleNextPage}
                        disabled={currentPage === totalPages}
                        className="px-3 py-1 text-sm bg-indigo-500 text-white rounded-md hover:bg-indigo-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                    >
                        Next
                    </button>
                </div>
            )}
            
            {/* Table Container - Added 'w-full' for robustness */}
            <div className="w-full max-h-96 overflow-y-auto rounded-lg border border-gray-300 dark:border-gray-600">
                {/* Table - Added 'table-auto' for better rendering */}
                <table className="min-w-full table-auto divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="sticky top-0 bg-gray-100 dark:bg-gray-700">
                        <tr>
                            {filteredColumns.map((col, index) => ( // <-- USE filteredColumns for headers
                                <th 
                                    key={index} 
                                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-400"
                                >
                                    {col}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-800 dark:divide-gray-700">
                        {rowsToDisplay.map((row, rowIndex) => ( // rowsToDisplay is already filtered
                            <tr 
                                key={startIndex + rowIndex}
                                className="hover:bg-indigo-50/50 dark:hover:bg-gray-700/50 transition-colors"
                            > 
                                {row.map((cell, cellIndex) => (
                                    <td 
                                        key={cellIndex} 
                                        className="px-6 py-3 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300"
                                    >
                                        {cell}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </motion.div>
    );
};

export default TableDisplay;