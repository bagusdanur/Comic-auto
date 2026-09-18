import { buildApp } from "./server.js";

const app = await buildApp();
await app.listen({
  host: process.env.TRANSLATOR_API_HOST || "127.0.0.1",
  port: Number(process.env.TRANSLATOR_API_PORT || 3210),
});
