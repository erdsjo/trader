import { getSimulationLogs, type EngineEvent } from "../api/client";
import { usePolling } from "../hooks/usePolling";

interface Props {
  simulationId: number | null;
}

const levelColor: Record<string, string> = {
  error: "text-red-400",
  warning: "text-yellow-400",
  info: "text-gray-400",
};

function EventRow({ event }: { event: EngineEvent }) {
  const time = new Date(event.timestamp).toLocaleTimeString();
  return (
    <div className={`flex gap-3 text-sm font-mono ${levelColor[event.level] ?? "text-gray-400"}`}>
      <span className="shrink-0 text-gray-500">{time}</span>
      <span className="shrink-0 uppercase w-14">{event.level}</span>
      <span className="break-all">{event.message}</span>
    </div>
  );
}

export function EngineActivity({ simulationId }: Props) {
  const { data, error } = usePolling(
    async () => {
      if (simulationId == null) return null;
      const res = await getSimulationLogs(simulationId, 50);
      return res.data;
    },
    5000,
    simulationId != null,
  );

  if (simulationId == null) return null;

  return (
    <div className="bg-gray-800 p-4 rounded-lg">
      <h2 className="text-lg font-semibold mb-3">Engine Activity</h2>

      {error && (
        <p className="text-gray-500 text-sm">
          No live logs available (simulation may not be running)
        </p>
      )}

      {data && data.events.length === 0 && (
        <p className="text-gray-500 text-sm">No events yet</p>
      )}

      {data && data.events.length > 0 && (
        <div className="max-h-64 overflow-y-auto space-y-1">
          {data.events.map((event, i) => (
            <EventRow key={i} event={event} />
          ))}
        </div>
      )}
    </div>
  );
}
