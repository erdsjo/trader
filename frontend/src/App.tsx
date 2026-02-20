import { useState } from "react";
import { getToken } from "./api/client";
import { Dashboard } from "./components/Dashboard";
import { LoginScreen } from "./components/LoginScreen";

function App() {
  const [authed, setAuthed] = useState(() => !!getToken());

  if (!authed) {
    return <LoginScreen onSuccess={() => setAuthed(true)} />;
  }

  return <Dashboard />;
}

export default App;
