import { getPortfolio } from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  simulationId: number | null;
}

export function Portfolio({ simulationId }: Props) {
  const { data: portfolio, loading } = usePolling(
    async () =>
      simulationId ? (await getPortfolio(simulationId)).data : null,
    10000,
    simulationId !== null
  );

  if (!simulationId) return <div className="bg-gray-800 p-4 rounded-lg">Select a simulation</div>;
  if (loading) return <div className="bg-gray-800 p-4 rounded-lg">Loading...</div>;
  if (!portfolio) return null;

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-3">Portfolio</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div>
          <div className="text-gray-400 text-sm">Total Value</div>
          <div className="text-xl font-bold">${portfolio.total_value.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
        </div>
        <div>
          <div className="text-gray-400 text-sm">Cash</div>
          <div className="text-xl">${portfolio.cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
        </div>
        <div>
          <div className="text-gray-400 text-sm">Total P/L</div>
          <div className={`text-xl font-bold ${portfolio.total_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
            ${portfolio.total_pnl.toFixed(2)} ({portfolio.total_pnl_pct.toFixed(2)}%)
          </div>
        </div>
      </div>

      {portfolio.positions.length > 0 && (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2">Symbol</th>
              <th className="text-right">Qty</th>
              <th className="text-right">Avg Price</th>
              <th className="text-right">Current</th>
              <th className="text-right">P/L</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.positions.map((p) => (
              <tr key={p.symbol} className="border-b border-gray-700">
                <td className="py-2 font-medium">{p.symbol}</td>
                <td className="text-right">{p.quantity}</td>
                <td className="text-right">${p.avg_price.toFixed(2)}</td>
                <td className="text-right">${p.current_price.toFixed(2)}</td>
                <td className={`text-right ${p.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                  ${p.pnl.toFixed(2)} ({p.pnl_pct.toFixed(1)}%)
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
