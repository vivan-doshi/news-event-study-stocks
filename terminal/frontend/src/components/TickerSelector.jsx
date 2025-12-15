import React from 'react';

const tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"];

const TickerSelector = ({ selected, onSelect }) => {
    // selected can be null or "ALL", treat null as "ALL" for UI
    const active = selected || "ALL";

    return (
        <div className="flex gap-2 mb-6 overflow-x-auto pb-2 scrollbar-hide">
            {tickers.map(t => (
                <button
                    key={t}
                    onClick={() => onSelect(t === "ALL" ? null : t)}
                    className={`
                        px-4 py-2 rounded-full font-bold text-sm tracking-wider transition-all duration-200
                        ${active === t
                            ? 'bg-terminal-green text-black shadow-[0_0_15px_rgba(0,255,65,0.4)] transform scale-105'
                            : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-white'}
                    `}
                >
                    {t}
                </button>
            ))}
        </div>
    );
};

export default TickerSelector;
