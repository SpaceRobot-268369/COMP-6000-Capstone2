import { useEffect, useState } from "react";

const apiBase = import.meta.env.VITE_API_URL || "";

export default function HomePage() {
  const [health, setHealth] = useState({ loading: true });

  useEffect(() => {
    fetch(`${apiBase}/api/health`)
      .then((response) => response.json())
      .then((data) => setHealth(data))
      .catch((error) => setHealth({ ok: false, message: String(error) }));
  }, []);

  return (
    <section>
      <h2>Home</h2>
      <p>Backend URL: {apiBase || "(not set)"}</p>
      <pre>{JSON.stringify(health, null, 2)}</pre>
    </section>
  );
}
