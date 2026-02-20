import { getTrades } from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  simulationId: number | null;
}

export function TradeHistory({ simulationId }: Props) {
  const { data: trades, loading } = usePolling(
    async () =>
      simulationId ? (await getTrades(simulationId)).data : null,
    10000,
    simulationId !== null
  );

  if (!simulationId) return null;
  if (loading) return <div className="bg-gray-800 p-4 rounded-lg">Loading trades...</div>;

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-3">Recent Trades</h2>
      {!trades || trades.length === 0 ? (
        <p className="text-gray-400">No trades yet</p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2">Time</th>
              <th className="text-left">Symbol</th>
              <th className="text-left">Side</th>
              <th className="text-right">Qty</th>
              <th className="text-right">Price</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={t.id} className="border-b border-gray-700">
                <td className="py-2">{new Date(t.timestamp).toLocaleString()}</td>
                <td>{t.symbol}</td>
                <td className={t.side === "buy" ? "text-green-400" : "text-red-400"}>
                  {t.side.toUpperCase()}
                </td>
                <td className="text-right">{t.quantity}</td>
                <td className="text-right">${t.price.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
