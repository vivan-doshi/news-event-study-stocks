
const HeroMetric = ({ value, label }) => {
    return (
        <div className="glass-panel flex flex-col items-center justify-center min-w-[300px]">
            <div className="text-terminal-green text-6xl font-bold tracking-tighter drop-shadow-[0_0_10px_rgba(0,255,65,0.5)]">
                {value}
            </div>
            <div className="text-gray-400 uppercase tracking-widest text-sm mt-2">{label}</div>
        </div>
    );
};

export default HeroMetric;
