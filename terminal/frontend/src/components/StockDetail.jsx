import React from 'react';

const StockDetail = ({ data }) => {
    // data = { ticker: "AAPL", has_shock: true, topics: [...] }
    if (!data) return <div className="text-gray-500">Select a Ticker to view details.</div>;

    return (
        <div className="h-full flex flex-col animate-fadeIn">
            <div className="flex justify-between items-center mb-6 border-b border-gray-700 pb-4">
                <div className="flex items-center gap-4">
                    <h2 className="text-5xl font-bold text-white tracking-tighter">{data.ticker}</h2>
                    {data.has_shock && (
                        <span className="px-3 py-1 bg-red-900/30 text-red-500 border border-red-500/50 rounded text-xs font-bold animate-pulse">
                            ACTIVE SHOCK
                        </span>
                    )}
                </div>
                <div className="text-right">
                    <div className="text-xs text-gray-500 uppercase tracking-widest mb-1">Total Topics</div>
                    <div className="text-2xl font-mono text-terminal-green">{data.topics.length}</div>
                </div>
            </div>

            {/* Detailed Topic List */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-2">
                {data.topics.map((t, idx) => {
                    const isShock = t.is_shock;
                    const isPos = t.z_score > 0;
                    let valColor = isShock ? (isPos ? "text-terminal-green" : "text-terminal-red") : "text-gray-400";
                    let borderClass = isShock ? (isPos ? "border-terminal-green/50" : "border-terminal-red/50") : "border-gray-800";

                    return (
                        <div key={idx} className={`glass-panel p-4 border ${borderClass} flex justify-between items-center group hover:bg-gray-800/50 transition-colors`}>
                            <div className="flex-1">
                                <h4 className="text-lg font-bold text-gray-200 mb-1">{t.name}</h4>
                                <p className="text-sm text-gray-500 font-mono tracking-tight">{t.desc}</p>
                            </div>

                            <div className="text-right pl-6 border-l border-gray-700 ml-6 w-32 flex-shrink-0">
                                <div className="text-xs text-gray-600 mb-1">Z-SCORE</div>
                                <div className={`text-2xl font-mono font-bold ${valColor}`}>
                                    {t.z_score > 0 ? "+" : ""}{t.z_score.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    );
};

export default StockDetail;
