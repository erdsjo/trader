import axios from "axios";

const api = axios.create({
  baseURL: "/api",
});

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
