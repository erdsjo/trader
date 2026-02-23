import { useState } from "react";
import type { Simulation } from "../api/client";
import { EngineActivity } from "./EngineActivity";
import { Portfolio } from "./Portfolio";
import { ProfitLoss } from "./ProfitLoss";
import { ScreenerResults } from "./ScreenerResults";
import { SimulationControl } from "./SimulationControl";
import { TradeHistory } from "./TradeHistory";

export function Dashboard() {
  const [selected, setSelected] = useState<Simulation | null>(null);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <h1 className="text-2xl font-bold mb-6">Trader Bot</h1>

      <div className="space-y-4">
        <SimulationControl selected={selected} onSelect={setSelected} />

        <EngineActivity simulationId={selected?.id ?? null} />

        <ScreenerResults simulationId={selected?.id ?? null} />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Portfolio simulationId={selected?.id ?? null} />
          <ProfitLoss simulationId={selected?.id ?? null} />
        </div>

        <TradeHistory simulationId={selected?.id ?? null} />
      </div>
    </div>
  );
}
