// Tests load from this checkout's models/ folder; the library itself never looks there.
import { fileURLToPath } from "node:url";
export const MODELS = fileURLToPath(new URL("../models/", import.meta.url));
