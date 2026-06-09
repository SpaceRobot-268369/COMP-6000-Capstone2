import pg from "pg";

const VALID_SLOTS = new Set([
  "layer_a",
  "layer_b",
  "layer_c",
  "layer_d",
  "layer_e_ambient",
  "layer_e_weather",
  "layer_e_events",
  "layer_e_aggregator",
]);

/**
 * Validates slot overrides against the live registry from Server B.
 * 
 * @param {Object} slots - Map of slot name to attempt ID.
 * @param {Array} registryLayers - List of layers from Server B registry.
 * @returns {Object} - Object with `valid` (boolean) and `errors` (object mapping slot to error message).
 */
export function validateSlots(slots, registryLayers) {
  const errors = {};
  
  for (const [slot, attemptId] of Object.entries(slots)) {
    if (!VALID_SLOTS.has(slot)) {
      errors[slot] = `Invalid slot name: '${slot}'`;
      continue;
    }
    
    // Empty attempt ID means revert to registry default (valid)
    if (!attemptId) {
      continue;
    }
    
    // Resolve the target layer ID and head filter
    let targetLayerId = slot;
    let targetHead = null;
    
    if (slot.startsWith("layer_e_")) {
      targetLayerId = "layer_e";
      targetHead = slot.slice("layer_e_".length); // "ambient" | "weather" | "events" | "aggregator"
    }
    
    const layer = registryLayers.find((l) => l.id === targetLayerId);
    if (!layer) {
      errors[slot] = `Layer '${targetLayerId}' not found in registry.`;
      continue;
    }
    
    const attempt = layer.attempts.find((a) => a.id === attemptId);
    if (!attempt) {
      errors[slot] = `Attempt '${attemptId}' not found in layer '${targetLayerId}'.`;
      continue;
    }
    
    if (targetHead && attempt.head !== targetHead) {
      errors[slot] = `Attempt '${attemptId}' head type is '${attempt.head}', but slot '${slot}' requires '${targetHead}'.`;
      continue;
    }
    
    if (!attempt.available) {
      errors[slot] = `Attempt '${attemptId}' is not available: ${attempt.unavailable_reason || "missing checkpoint weights"}`;
      continue;
    }
  }
  
  return {
    valid: Object.keys(errors).length === 0,
    errors,
  };
}

/**
 * Retrieves the active global model configuration slots from PostgreSQL.
 * 
 * @param {pg.Pool} pool - PG pool.
 * @returns {Promise<Object>} - Object with `slots` object mapping slot names to attempt IDs.
 */
export async function getActiveConfig(pool) {
  const query = `
    SELECT s.slot, s.attempt_id
    FROM model_config_slots s
    JOIN model_configs c ON s.config_id = c.id
    WHERE c.is_active = TRUE AND c.user_id IS NULL;
  `;
  try {
    const { rows } = await pool.query(query);
    const slots = {};
    for (const row of rows) {
      slots[row.slot] = row.attempt_id;
    }
    return { slots };
  } catch (err) {
    console.error("Failed to query model config slots:", err);
    return { slots: {} };
  }
}

/**
 * Persists the selected slots to PostgreSQL under the active global config.
 * 
 * @param {pg.Pool} pool - PG pool.
 * @param {Object} slots - Map of slot name to attempt ID.
 * @param {Array} registryLayers - List of layers from Server B registry.
 * @returns {Promise<Object>} - The updated active config slots.
 */
export async function setSlots(pool, slots, registryLayers) {
  const { valid, errors } = validateSlots(slots, registryLayers);
  if (!valid) {
    const err = new Error("Validation failed");
    err.errors = errors;
    err.status = 400;
    throw err;
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // 1. Get or create active global config row
    const upsertConfigQuery = `
      INSERT INTO model_configs (user_id, name, is_active)
      VALUES (NULL, 'default', TRUE)
      ON CONFLICT (is_active) WHERE user_id IS NULL AND is_active
      DO UPDATE SET updated_at = NOW()
      RETURNING id;
    `;
    const { rows } = await client.query(upsertConfigQuery);
    const configId = rows[0].id;

    // 2. Delete all existing slots for this config
    await client.query("DELETE FROM model_config_slots WHERE config_id = $1", [configId]);

    // 3. Insert new overrides
    const insertSlotQuery = `
      INSERT INTO model_config_slots (config_id, slot, attempt_id)
      VALUES ($1, $2, $3);
    `;
    
    for (const [slot, attemptId] of Object.entries(slots)) {
      if (attemptId) {
        await client.query(insertSlotQuery, [configId, slot, attemptId]);
      }
    }

    await client.query("COMMIT");
    
    // Return the fresh persisted slot map
    return getActiveConfig(pool);
  } catch (err) {
    await client.query("ROLLBACK");
    console.error("Failed to save model config slots:", err);
    throw err;
  } finally {
    client.release();
  }
}
