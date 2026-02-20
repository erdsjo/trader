import { useState } from "react";
import {
  createSimulation,
  deleteSimulation,
  getSimulations,
  startSimulation,
  stopSimulation,
  type Simulation,
} from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  selected: Simulation | null;
  onSelect: (sim: Simulation | null) => void;
}

export function SimulationControl({ selected, onSelect }: Props) {
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [capital, setCapital] = useState("10000");
  const [symbols, setSymbols] = useState("AAPL");
  const [interval, setInterval] = useState("1d");
  const [tickSeconds, setTickSeconds] = useState("60");
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);

  const { data: simulations } = usePolling(
    async () => (await getSimulations()).data,
    5000
  );

  const handleCreate = async () => {
    setError("");
    try {
      const sim = await createSimulation({
        name,
        initial_capital: parseFloat(capital),
        config: {
          symbols: symbols.split(",").map((s) => s.trim()),
          interval,
          tick_seconds: parseFloat(tickSeconds),
        },
      });
      onSelect(sim.data);
      setShowCreate(false);
      setName("");
    } catch {
      setError("Failed to create simulation");
    }
  };

  const handleStart = async () => {
    if (selected) {
      setError("");
      setStarting(true);
      try {
        await startSimulation(selected.id);
      } catch (err: unknown) {
        const detail =
          err && typeof err === "object" && "response" in err
            ? (err as { response?: { data?: { detail?: string } } }).response
                ?.data?.detail
            : undefined;
        setError(detail || "Failed to start simulation");
      } finally {
        setStarting(false);
      }
    }
  };

  const handleStop = async () => {
    if (selected) {
      setError("");
      try {
        await stopSimulation(selected.id);
      } catch {
        setError("Failed to stop simulation");
      }
    }
  };

  const handleDelete = async () => {
    if (selected && confirm(`Delete "${selected.name}"? This cannot be undone.`)) {
      setError("");
      try {
        await deleteSimulation(selected.id);
        onSelect(null);
      } catch {
        setError("Failed to delete simulation");
      }
    }
  };

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <div className="flex items-center gap-4 flex-wrap">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Simulation</label>
          <select
            className="bg-gray-700 text-white px-3 py-2 rounded"
            value={selected?.id ?? ""}
            onChange={(e) => {
              const sim = simulations?.find(
                (s) => s.id === Number(e.target.value)
              );
              if (sim) onSelect(sim);
            }}
          >
            <option value="">Select simulation...</option>
            {simulations?.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} (${s.initial_capital.toLocaleString()})
              </option>
            ))}
          </select>
        </div>

        {selected && (
          <span
            className={`px-2 py-1 rounded text-sm self-end mb-1 ${
              selected.status === "running"
                ? "bg-green-600"
                : selected.status === "stopped"
                  ? "bg-red-600"
                  : "bg-yellow-600"
            }`}
          >
            {selected.status}
          </span>
        )}

        {selected && selected.status !== "running" && (
          <button
            onClick={handleStart}
            disabled={starting}
            className="bg-green-600 hover:bg-green-700 disabled:opacity-50 px-4 py-2 rounded self-end"
          >
            {starting ? "Starting..." : "Start"}
          </button>
        )}
        {selected && selected.status === "running" && (
          <button
            onClick={handleStop}
            className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded self-end"
          >
            Stop
          </button>
        )}

        {selected && selected.status !== "running" && (
          <button
            onClick={handleDelete}
            className="bg-gray-600 hover:bg-red-700 px-4 py-2 rounded self-end"
          >
            Delete
          </button>
        )}

        <button
          onClick={() => setShowCreate(!showCreate)}
          className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded ml-auto self-end"
        >
          {showCreate ? "Cancel" : "New Simulation"}
        </button>
      </div>

      {error && (
        <p className="text-red-400 text-sm mt-2">{error}</p>
      )}

      {showCreate && (
        <div className="mt-4 border-t border-gray-700 pt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Simulation Name
            </label>
            <input
              placeholder="e.g. Tech Stocks Daily"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="bg-gray-700 px-3 py-2 rounded w-full text-white"
            />
            <p className="text-xs text-gray-500 mt-1">
              A descriptive name to identify this simulation
            </p>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Initial Capital ($)
            </label>
            <input
              type="number"
              placeholder="10000"
              value={capital}
              onChange={(e) => setCapital(e.target.value)}
              className="bg-gray-700 px-3 py-2 rounded w-full text-white"
              min="0"
              step="1000"
            />
            <p className="text-xs text-gray-500 mt-1">
              Starting cash balance for the simulated portfolio
            </p>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Symbols
            </label>
            <input
              placeholder="e.g. AAPL, MSFT, GOOGL"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              className="bg-gray-700 px-3 py-2 rounded w-full text-white"
            />
            <p className="text-xs text-gray-500 mt-1">
              Comma-separated stock tickers to trade (Yahoo Finance)
            </p>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Data Interval
            </label>
            <select
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
              className="bg-gray-700 px-3 py-2 rounded w-full text-white"
            >
              <option value="1m">1 Minute</option>
              <option value="5m">5 Minutes</option>
              <option value="15m">15 Minutes</option>
              <option value="1h">1 Hour</option>
              <option value="1d">1 Day</option>
              <option value="1wk">1 Week</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Candlestick interval for price data from Yahoo Finance
            </p>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">
              Tick Interval (seconds)
            </label>
            <input
              type="number"
              placeholder="60"
              value={tickSeconds}
              onChange={(e) => setTickSeconds(e.target.value)}
              className="bg-gray-700 px-3 py-2 rounded w-full text-white"
              min="1"
              step="1"
            />
            <p className="text-xs text-gray-500 mt-1">
              How often the engine fetches data and evaluates trades
            </p>
          </div>

          <div className="flex items-end">
            <button
              onClick={handleCreate}
              disabled={!name || !capital || !symbols}
              className="bg-green-600 hover:bg-green-700 disabled:opacity-50 px-6 py-2 rounded w-full font-medium"
            >
              Create Simulation
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
