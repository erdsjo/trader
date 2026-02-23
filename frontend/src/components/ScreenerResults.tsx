import { usePolling } from '../hooks/usePolling';
import { getScreeningResults, type ScreeningResultItem } from '../api/client';

interface Props {
  simulationId: number | null;
}

export function ScreenerResults({ simulationId }: Props) {
  const { data: results } = usePolling<ScreeningResultItem[]>(
    async () => simulationId ? await getScreeningResults(simulationId) : [],
    10000,
    !!simulationId,
  );

  if (!simulationId || !results || results.length === 0) {
    return null;
  }

  const bySector: Record<string, ScreeningResultItem[]> = {};
  for (const r of results) {
    (bySector[r.sector] ??= []).push(r);
  }

  const selectedCount = results.filter(r => r.selected).length;

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-3">
        Stock Screener
        <span className="text-sm font-normal text-gray-400 ml-2">
          {selectedCount} selected / {results.length} candidates
        </span>
      </h2>
      <div className="space-y-3">
        {Object.entries(bySector).sort(([a], [b]) => a.localeCompare(b)).map(([sector, stocks]) => (
          <div key={sector}>
            <h3 className="text-sm text-gray-400 mb-1">{sector}</h3>
            <div className="flex flex-wrap gap-1">
              {stocks
                .sort((a, b) => (b.opportunity_score ?? b.volatility) - (a.opportunity_score ?? a.volatility))
                .map(s => (
                  <span
                    key={s.symbol}
                    className={`text-xs px-2 py-1 rounded ${
                      s.selected
                        ? 'bg-green-900 text-green-300 font-medium'
                        : 'bg-gray-700 text-gray-400'
                    }`}
                    title={`Vol: ${(s.volume_avg / 1e6).toFixed(1)}M | Volatility: ${(s.volatility * 100).toFixed(1)}%`}
                  >
                    {s.symbol}
                  </span>
                ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
