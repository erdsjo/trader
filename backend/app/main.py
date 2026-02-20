from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import portfolio, simulation, strategy, trades
from app.config import settings

app = FastAPI(title="Trader Bot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation.router, prefix=settings.api_prefix)
app.include_router(portfolio.router, prefix=settings.api_prefix)
app.include_router(trades.router, prefix=settings.api_prefix)
app.include_router(strategy.router, prefix=settings.api_prefix)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
