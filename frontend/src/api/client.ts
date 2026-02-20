import axios from "axios";

const TOKEN_KEY = "auth_token";

export const getToken = (): string | null => localStorage.getItem(TOKEN_KEY);
export const setToken = (token: string): void =>
  localStorage.setItem(TOKEN_KEY, token);
export const clearToken = (): void => localStorage.removeItem(TOKEN_KEY);

const api = axios.create({
  baseURL: "/api",
});

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearToken();
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

export interface Simulation {
  id: number;
  name: string;
  initial_capital: number;
  current_cash: number;
  start_time: string;
  status: string;
  config: Record<string, unknown>;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  pnl_pct: number;
}

export interface Portfolio {
  total_value: number;
  cash: number;
  positions: Position[];
  total_pnl: number;
  total_pnl_pct: number;
}

export interface Trade {
  id: number;
  simulation_id: number;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  timestamp: string;
  strategy: string | null;
}

export interface PerformancePoint {
  timestamp: string;
  total_value: number;
}

export interface Performance {
  points: PerformancePoint[];
  daily_pnl: number;
  total_pnl: number;
  total_pnl_pct: number;
}

export interface EngineEvent {
  timestamp: string;
  level: "info" | "warning" | "error";
  message: string;
}

export interface SimulationLogs {
  simulation_id: number;
  events: EngineEvent[];
}

export const getSimulations = () => api.get<Simulation[]>("/simulations");
export const createSimulation = (data: {
  name: string;
  initial_capital: number;
  config?: Record<string, unknown>;
}) => api.post<Simulation>("/simulations", data);
export const startSimulation = (id: number) =>
  api.post(`/simulations/${id}/start`);
export const stopSimulation = (id: number) =>
  api.post(`/simulations/${id}/stop`);
export const getPortfolio = (id: number) =>
  api.get<Portfolio>(`/simulations/${id}/portfolio`);
export const getTrades = (id: number) =>
  api.get<Trade[]>(`/simulations/${id}/trades`);
export const getPerformance = (id: number) =>
  api.get<Performance>(`/simulations/${id}/performance`);
export const getSimulationLogs = (id: number, limit: number = 50) =>
  api.get<SimulationLogs>(`/simulations/${id}/logs`, { params: { limit } });

export const login = async (password: string): Promise<string> => {
  const { data } = await api.post<{ token: string }>("/auth/login", {
    password,
  });
  return data.token;
};
