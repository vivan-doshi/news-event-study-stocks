import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const EquityChart = ({ data, selectedTicker }) => {
    // data is an array of objects: { date, strategy, benchmark, AAPL, MSFT... }

    if (!data || data.length === 0) return <div className="h-full flex items-center justify-center text-gray-600">Loading Chart Data...</div>;

    // Determine what to show
    const showStock = selectedTicker && selectedTicker !== "STRATEGY";
    const lineKey = showStock ? selectedTicker : "strategy";
    const lineColor = showStock ? "#3b82f6" : "#00ff41"; // Blue for stock, Green for strategy
    const title = showStock ? `PRICE HISTORY: ${selectedTicker} vs BENCHMARK` : "CUMULATIVE RETURNS (STRATEGY vs BENCHMARK)";

    return (
        <div className="w-full h-full flex flex-col">
            <div className="flex justify-between items-center mb-2 px-2">
                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest">{title}</h3>
                <div className="flex gap-4 text-[10px] uppercase">
                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ backgroundColor: lineColor }}></span> {showStock ? selectedTicker : "Strategy"}</span>
                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-600"></span> Benchmark</span>
                </div>
            </div>

            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorStrategy" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={lineColor} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={lineColor} stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="colorBench" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#4b5563" stopOpacity={0.1} />
                                <stop offset="95%" stopColor="#4b5563" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                        <XAxis
                            dataKey="date"
                            stroke="#6b7280"
                            tick={{ fontSize: 10 }}
                            tickMargin={10}
                            interval="preserveStartEnd"
                            label={{ value: "Date", position: "insideBottomRight", offset: -5, fill: "#6b7280", fontSize: 10 }}
                        />
                        <YAxis
                            stroke="#6b7280"
                            tick={{ fontSize: 10 }}
                            orientation="right"
                            domain={['auto', 'auto']}
                            label={{ value: "Price ($)", angle: -90, position: "insideRight", offset: 10, fill: "#6b7280", fontSize: 10 }}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#000', borderColor: '#333', color: '#fff' }}
                            itemStyle={{ fontSize: '12px' }}
                            labelStyle={{ color: '#9ca3af', marginBottom: '5px' }}
                        />

                        <Area
                            type="monotone"
                            dataKey="benchmark"
                            stroke="#4b5563"
                            fillOpacity={1}
                            fill="url(#colorBench)"
                            strokeWidth={1}
                            dot={false}
                        />
                        <Area
                            type="monotone"
                            dataKey={lineKey}
                            stroke={lineColor}
                            fillOpacity={1}
                            fill="url(#colorStrategy)"
                            strokeWidth={2}
                            dot={false} // Remove dots for cleaner look
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default EquityChart;
