from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from logic import get_intraday_signals, get_overnight_signal, get_portfolio_stats, get_historical_equity

app = FastAPI(title="Mag7 Sortino-Maximizer")

# Allow CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Mag7 Alpha Squad Terminal Online", "market_regime": "Risk-On"}

@app.get("/intraday/sniper")
def sniper_signal():
    """
    Returns the Topic Shocks for Intraday Trading.
    Model: FF3_TopicShock (Sortino 49.5)
    """
    return {
        "strategy": "Intraday Topic Sniper",
        "signals": get_intraday_signals()
    }

@app.get("/overnight/swing")
def swing_signal():
    """
    Returns Overnight Signals for Global + Mag7.
    """
    return {
        "strategy": "Overnight Swing",
        "data": get_overnight_signal()
    }

@app.get("/stats")
def portfolio_stats():
    """
    Returns the core dashboard metrics (Sortino, Sharpe, etc.)
    """
    return get_portfolio_stats()

@app.get("/charts/equity")
def equity_curve():
    """
    Returns historical equity curve for plotting.
    """
    return {"data": get_historical_equity()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
