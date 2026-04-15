import cors from "cors";
import express from "express";
import pg from "pg";

const app = express();
const port = Number(process.env.PORT || 4000);

app.use(cors());
app.use(express.json());

app.get("/api/health", async (_req, res) => {
  const client = new pg.Client({ connectionString: process.env.DATABASE_URL });

  try {
    await client.connect();
    const { rows } = await client.query("SELECT NOW() AS now");
    await client.end();
    res.json({ ok: true, db: "connected", now: rows[0].now });
  } catch (error) {
    res.status(500).json({ ok: false, db: "error", message: String(error.message || error) });
  }
});

app.get("/", (_req, res) => {
  res.json({ service: "backend", status: "running" });
});

app.listen(port, "0.0.0.0", () => {
  console.log(`Backend running on http://0.0.0.0:${port}`);
});
