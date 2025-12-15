import React from 'react';

const StockCard = ({ data, isSelected, onClick }) => {
    // data = { ticker: "AAPL", has_shock: true, topics: [...] }

    return (
        <div
            onClick={() => onClick(data.ticker)}
            className={`
            glass-panel p-4 flex flex-col h-full transition-all duration-300 cursor-pointer relative overflow-hidden
            ${isSelected ? 'border-terminal-green shadow-[0_0_20px_rgba(0,255,65,0.15)] bg-green-900/10' :
                    data.has_shock ? 'border-gray-600' : 'border-gray-800 opacity-80'}
            hover:bg-gray-800 hover:border-gray-500 hover:opacity-100
        `}>
            {isSelected && <div className="absolute top-0 left-0 w-1 h-full bg-terminal-green"></div>}

            {/* Header */}
            <div className="flex justify-between items-center mb-4 border-b border-gray-700 pb-2">
                <span className={`text-3xl font-bold tracking-tighter ${isSelected ? 'text-terminal-green' : 'text-white'}`}>
                    {data.ticker}
                </span>
                {data.has_shock && <div className="w-2 h-2 rounded-full bg-terminal-green animate-pulse shadow-[0_0_10px_#00ff41]"></div>}
            </div>

            {/* Topics List - Improved Readability */}
            <div className="space-y-2.5 flex-1">
                {data.topics.map((t, idx) => {
                    const isShock = t.is_shock;
                    const isPos = t.z_score > 0;

                    // Color Logic
                    let valColor = "text-gray-500";
                    if (isShock) valColor = isPos ? "text-terminal-green font-bold text-shadow-green" : "text-terminal-red font-bold text-shadow-red";

                    // Bg highlight for strong signals
                    let rowBg = isShock ? (isPos ? "bg-green-900/20" : "bg-red-900/20") : "bg-transparent";

                    return (
                        <div key={idx} className={`flex justify-between items-center text-xs p-1.5 rounded ${rowBg} group relative`}>
                            {/* Tooltip on Hover */}
                            <div className="absolute left-0 -top-8 bg-black border border-gray-700 p-2 rounded text-[10px] text-gray-300 w-48 shadow-xl opacity-0 group-hover:opacity-100 pointer-events-none z-20 transition-opacity">
                                <span className="font-bold text-white block mb-0.5">{t.name}</span>
                                {t.desc}
                            </div>

                            {/* Topic Name: Allow more width, less truncation */}
                            <span className="text-gray-300 font-medium truncate flex-1 pr-2" title={t.name}>
                                {t.name}
                            </span>

                            {/* Score: Fixed width to align right */}
                            <span className={`font-mono ${valColor} w-16 text-right`}>
                                {t.z_score > 0 ? "+" : ""}{t.z_score.toFixed(2)} σ
                            </span>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}

const TopicHeatmap = ({ stocks, selectedTicker, onSelect }) => {
    if (!stocks || !Array.isArray(stocks)) return <div className="text-gray-500 animate-pulse p-4">Loading Matrix Data...</div>;

    return (
        <div className="w-full h-full flex flex-col">
            <div className="flex justify-between items-center mb-4 border-b border-gray-700 pb-2 flex-shrink-0">
                <h3 className="text-gray-400 text-sm uppercase tracking-wider">Mag7 Intraday Matrix</h3>
                <span className="text-[10px] text-terminal-green animate-pulse">● LIVE FEED</span>
            </div>

            {/* Grid Layout - Adjusted columns for readability */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4 overflow-y-auto pr-2 pb-2">
                {stocks.map((stock, idx) => (
                    <StockCard
                        key={idx}
                        data={stock}
                        isSelected={selectedTicker === stock.ticker}
                        onClick={onSelect}
                    />
                ))}
            </div>
        </div>
    );
};

export default TopicHeatmap;
