import { useState, useEffect } from 'react'
import HeroMetric from './components/HeroMetric'
import TopicHeatmap from './components/TopicHeatmap'
import EquityChart from './components/EquityChart'
import TickerSelector from './components/TickerSelector'
import StockDetail from './components/StockDetail'
import { Activity, Terminal } from 'lucide-react'

function App() {
  const [stats, setStats] = useState(null)
  const [intraday, setIntraday] = useState([])
  const [overnight, setOvernight] = useState(null)
  const [equityData, setEquityData] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedTicker, setSelectedTicker] = useState("AAPL") // Default to AAPL

  useEffect(() => {
    const fetchData = async () => {
      try {
        const statsRes = await fetch('http://localhost:8000/stats')
        const intradayRes = await fetch('http://localhost:8000/intraday/sniper')
        const overnightRes = await fetch('http://localhost:8000/overnight/swing')
        const equityRes = await fetch('http://localhost:8000/charts/equity')

        setStats(await statsRes.json())
        const iData = await intradayRes.json()
        setIntraday(iData.signals)
        const oData = await overnightRes.json()
        setOvernight(oData.data) // Now a dict { 'GLOBAL': {..}, 'AAPL': {..} }
        const eData = await equityRes.json()
        setEquityData(eData.data)

        setLoading(false)
      } catch (error) {
        console.error("Failed to fetch terminal data", error)
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [])

  if (loading || !stats) return <div className="h-screen flex flex-col items-center justify-center text-terminal-green bg-terminal-black"><Activity className="animate-spin mb-4" />BOOTING MAG7 ALPHA PROTOCOL...</div>

  // Helper to get current overnight signal
  const currentOvernight = overnight ? (selectedTicker ? overnight[selectedTicker] : overnight['GLOBAL']) : null;
  // Helper to find stock data for details
  const currentStockData = selectedTicker ? intraday.find(s => s.ticker === selectedTicker) : null;

  return (
    <div className="min-h-screen p-6 max-w-[1600px] mx-auto bg-terminal-black text-gray-300 font-mono">
      {/* Header */}
      <header className="flex justify-between items-center mb-6 border-b border-gray-800 pb-4">
        <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => setSelectedTicker(null)}>
          <Terminal className="text-terminal-green" size={24} />
          <h1 className="text-2xl font-bold tracking-tight text-white">
            MAG7<span className="text-terminal-green text-opacity-80"> // </span>SORTINO MAXIMIZER
          </h1>
        </div>

        {/* Ticker Selector in Header Area */}
        <div className="flex-1 px-8 flex justify-center">
          <TickerSelector selected={selectedTicker} onSelect={setSelectedTicker} />
        </div>

        <div className="flex items-center gap-6 text-xs uppercase tracking-widest flex-shrink-0">
          <div className="flex flex-col text-right">
            <span className="text-gray-500">Market Regime</span>
            <span className="text-terminal-green font-bold">Risk-On</span>
          </div>
          <div className="flex flex-col text-right">
            <span className="text-white flex items-center justify-end gap-2"><span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> ONLINE</span>
          </div>
        </div>
      </header>

      {/* Grid Layout */}
      <div className="grid grid-cols-12 gap-6 h-[calc(100vh-140px)]">

        {/* Left Column: Metrics & Equity Curve (Span 8) */}
        <div className="col-span-12 lg:col-span-8 flex flex-col h-full gap-4">

          {/* Top Row: Metrics */}
          <div className="grid grid-cols-4 gap-4 flex-shrink-0">
            <div className="col-span-2">
              {stats && <HeroMetric label="SORTINO RATIO" value={stats.sortino.toFixed(2)} isGood={true} />}
            </div>
            <div className="glass-panel flex flex-col justify-center p-4">
              <span className="text-gray-500 text-xs font-bold font-mono">PROFIT FACTOR</span>
              <span className="text-2xl font-bold text-white font-mono">{stats ? stats.profit_factor : "..."}</span>
            </div>
            <div className="glass-panel flex flex-col justify-center p-4">
              <span className="text-gray-500 text-xs font-bold font-mono">SHARPE RATIO</span>
              <span className="text-2xl font-bold text-white font-mono">{stats ? stats.sharpe : "..."}</span>
            </div>
          </div>

          {/* Middle: Equity Chart (Flex Grow) */}
          <div className="flex-1 min-h-0 bg-terminal-gray/30 rounded-lg p-2 transition-all duration-500">
            <EquityChart data={equityData} selectedTicker={selectedTicker} />
          </div>

          {/* Bottom: Overnight Signal (Dynamic based on Ticker) */}
          <div className="glass-panel p-6 flex justify-between items-center flex-shrink-0 transition-colors duration-300 border-l-4 border-terminal-green">
            <div>
              <h3 className="text-gray-500 text-xs font-bold mb-1">
                {selectedTicker ? (selectedTicker + " OVERNIGHT SIGNAL") : "GLOBAL OVERNIGHT SIGNAL"} (T+1)
              </h3>
              <div className="text-xl font-bold text-white">
                Sentiment Z: <span className={currentOvernight?.score > 0 ? "text-terminal-green" : "text-terminal-red"}>
                  {currentOvernight?.score}
                </span>
              </div>
              <div className="mt-3 text-xs text-gray-300 leading-normal font-medium border-l-2 border-terminal-green pl-3 py-1">
                {currentOvernight?.reasoning && `"${currentOvernight.reasoning}"`}
              </div>
            </div>
            <div className="text-right relative group cursor-help">
              <div className="text-[10px] text-gray-500 mb-1 border-b border-dotted border-gray-500 inline-block">RECOMMENDATION (?)</div>
              {/* Tooltip */}
              <div className="hidden group-hover:block absolute bottom-full right-0 mb-2 w-64 p-3 bg-gray-900 border border-gray-700 rounded shadow-2xl z-50 text-left">
                <h4 className="text-xs font-bold text-gray-300 mb-2 border-b border-gray-800 pb-1">SIGNAL LEGEND</h4>
                <ul className="text-[10px] space-y-2">
                  <li className="flex justify-between"><span className="text-terminal-green font-bold">BUY</span> <span className="text-gray-400">Positive sentiment momentum</span></li>
                  <li className="flex justify-between"><span className="text-terminal-red font-bold">SELL</span> <span className="text-gray-400">Negative shock detected</span></li>
                  <li className="flex justify-between"><span className="text-yellow-500 font-bold">HOLD</span> <span className="text-gray-400">Low conviction / Mixed signals</span></li>
                  <li className="flex justify-between"><span className="text-blue-400 font-bold">CASH</span> <span className="text-gray-400">High Uncertainty / Volatility</span></li>
                </ul>
              </div>
              <div className="text-2xl font-bold tracking-widest animate-pulse text-white">
                {currentOvernight?.action}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Intraday Matrix OR Detail View (Span 4) */}
        <div className="col-span-12 lg:col-span-4 flex flex-col h-full overflow-hidden">
          {selectedTicker ? (
            <StockDetail data={currentStockData} />
          ) : (
            <TopicHeatmap
              stocks={intraday}
              selectedTicker={selectedTicker}
              onSelect={setSelectedTicker}
            />
          )}
        </div>

      </div>
    </div>
  )
}
export default App
