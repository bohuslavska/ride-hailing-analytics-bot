-- Applied after the bulk COPY: building these up front makes the load roughly
-- twice as slow, because every one of the 739k rides would have to be inserted
-- into each index individually.

CREATE INDEX rides_origin_zone_idx   ON rides (origin_zone_id);
CREATE INDEX rides_calculated_at_idx ON rides (calculated_at);
CREATE INDEX rides_final_status_idx  ON rides (final_status);
CREATE INDEX rides_user_idx          ON rides (user_id);
CREATE INDEX zone_state_zone_idx     ON zone_state (zone_id);
CREATE INDEX zone_state_ts_idx       ON zone_state (ts);

-- Denormalised view: this is the relation the bot is told to query. It
-- pre-joins the zone descriptions and derives the calendar columns, which
-- removes the two things an LLM most often gets wrong writing SQL unaided.
CREATE VIEW rides_enriched AS
SELECT
    r.ride_id,
    r.user_id,
    r.calculated_at,
    CAST(r.calculated_at AS DATE)                    AS ride_date,
    CAST(EXTRACT(HOUR FROM r.calculated_at) AS INT)  AS hour,
    CAST(EXTRACT(DOW  FROM r.calculated_at) AS INT)  AS day_of_week,
    r.is_weekend,
    r.is_peak_hour,
    r.curfew,
    r.air_raid_alert,
    r.weather,
    r.special_event,

    r.origin_zone_id,
    origin.zone_name                                 AS origin_zone_name,
    origin.zone_type                                 AS origin_zone_type,
    r.destination_zone_id,
    destination.zone_name                            AS destination_zone_name,
    destination.zone_type                            AS destination_zone_type,

    r.distance_km,
    r.estimated_duration_minutes,
    r.eta_minutes,
    r.traffic_index,
    r.demand_count,
    r.available_drivers,
    r.demand_supply_ratio,

    r.base_price,
    r.surge_multiplier,
    r.final_price,

    r.placed,
    r.accepted,
    r.churned_to_competitor,
    r.search_wait_minutes,
    r.final_status,
    r.placed_at,
    r.accepted_at
FROM rides r
JOIN zones origin      ON origin.zone_id      = r.origin_zone_id
JOIN zones destination ON destination.zone_id = r.destination_zone_id;

COMMENT ON VIEW rides_enriched IS
    'rides pre-joined to origin and destination zone descriptions, with calendar columns derived.';
