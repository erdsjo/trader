import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getPerformance } from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  simulationId: number | null;
}

export function ProfitLoss({ simulationId }: Props) {
  const { data: performance, loading } = usePolling(
    async () =>
      simulationId ? (await getPerformance(simulationId)).data : null,
    10000,
    simulationId !== null
  );

  if (!simulationId) return null;
  if (loading) return <div className="bg-gray-800 p-4 rounded-lg">Loading chart...</div>;
  if (!performance || performance.points.length === 0) {
    return <div className="bg-gray-800 p-4 rounded-lg">No performance data yet</div>;
  }

  const chartData = performance.points.map((p) => ({
    time: new Date(p.timestamp).toLocaleDateString(),
    value: p.total_value,
  }));

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-3">Performance</h2>
      <div className="flex gap-4 mb-4 text-sm">
        <div>
          <span className="text-gray-400">Daily P/L: </span>
          <span className={performance.daily_pnl >= 0 ? "text-green-400" : "text-red-400"}>
            ${performance.daily_pnl.toFixed(2)}
          </span>
        </div>
        <div>
          <span className="text-gray-400">Total P/L: </span>
          <span className={performance.total_pnl >= 0 ? "text-green-400" : "text-red-400"}>
            ${performance.total_pnl.toFixed(2)} ({performance.total_pnl_pct.toFixed(2)}%)
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#9CA3AF" />
          <YAxis stroke="#9CA3AF" />
          <Tooltip
            contentStyle={{ backgroundColor: "#1F2937", border: "none" }}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#10B981"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>

      {performance.sector_pnl && Object.keys(performance.sector_pnl).length > 0 && (
        <div className="mt-4 border-t border-gray-700 pt-3">
          <h3 className="text-sm font-medium text-gray-400 mb-2">P/L by Sector</h3>
          <div className="space-y-1">
            {Object.entries(performance.sector_pnl)
              .sort(([, a], [, b]) => b - a)
              .map(([sector, pnl]) => (
                <div key={sector} className="flex justify-between text-sm">
                  <span className="text-gray-300">{sector}</span>
                  <span className={pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                    ${pnl.toFixed(2)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
