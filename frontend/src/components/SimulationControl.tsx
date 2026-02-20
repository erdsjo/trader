import { useState } from "react";
import {
  createSimulation,
  getSimulations,
  startSimulation,
  stopSimulation,
  type Simulation,
} from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  selected: Simulation | null;
  onSelect: (sim: Simulation) => void;
}

export function SimulationControl({ selected, onSelect }: Props) {
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [capital, setCapital] = useState("10000");
  const [symbols, setSymbols] = useState("AAPL");

  const { data: simulations } = usePolling(
    async () => (await getSimulations()).data,
    5000
  );

  const handleCreate = async () => {
    const sim = await createSimulation({
      name,
      initial_capital: parseFloat(capital),
      config: {
        symbols: symbols.split(",").map((s) => s.trim()),
        interval: "1d",
      },
    });
    onSelect(sim.data);
    setShowCreate(false);
    setName("");
  };

  const handleStart = async () => {
    if (selected) {
      await startSimulation(selected.id);
    }
  };

  const handleStop = async () => {
    if (selected) {
      await stopSimulation(selected.id);
    }
  };

  return (
    <div className="bg-gray-800 p-4 rounded-lg flex items-center gap-4 flex-wrap">
      <select
        className="bg-gray-700 text-white px-3 py-2 rounded"
        value={selected?.id ?? ""}
        onChange={(e) => {
          const sim = simulations?.find((s) => s.id === Number(e.target.value));
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

      {selected && (
        <span
          className={`px-2 py-1 rounded text-sm ${
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
          className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
        >
          Start
        </button>
      )}
      {selected && selected.status === "running" && (
        <button
          onClick={handleStop}
          className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded"
        >
          Stop
        </button>
      )}

      <button
        onClick={() => setShowCreate(!showCreate)}
        className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded ml-auto"
      >
        New Simulation
      </button>

      {showCreate && (
        <div className="w-full flex gap-2 mt-2">
          <input
            placeholder="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="bg-gray-700 px-3 py-2 rounded flex-1"
          />
          <input
            placeholder="Capital"
            value={capital}
            onChange={(e) => setCapital(e.target.value)}
            className="bg-gray-700 px-3 py-2 rounded w-32"
          />
          <input
            placeholder="Symbols (comma sep)"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            className="bg-gray-700 px-3 py-2 rounded flex-1"
          />
          <button
            onClick={handleCreate}
            className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
          >
            Create
          </button>
        </div>
      )}
    </div>
  );
}
