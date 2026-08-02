-- Analytical schema for the synthetic ride-hailing dataset.
--
-- The tables stay normalised so that the funnel constraints can be declared and
-- enforced by the database rather than trusted from the generator. Secondary
-- indexes and the reporting view live in sql/indexes.sql and are applied after
-- the bulk COPY.

DROP VIEW IF EXISTS rides_enriched;
DROP TABLE IF EXISTS rides;
DROP TABLE IF EXISTS zone_state;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS zones;

CREATE TABLE zones (
    zone_id                 TEXT PRIMARY KEY,
    zone_name               TEXT             NOT NULL,
    zone_type               TEXT             NOT NULL,
    latitude                DOUBLE PRECISION NOT NULL,
    longitude               DOUBLE PRECISION NOT NULL,
    distance_from_center_km DOUBLE PRECISION NOT NULL,
    base_demand             DOUBLE PRECISION NOT NULL,
    base_supply             DOUBLE PRECISION NOT NULL,
    base_traffic            DOUBLE PRECISION NOT NULL,
    base_eta_minutes        DOUBLE PRECISION NOT NULL
);

COMMENT ON TABLE zones IS
    'One row per geographic zone. 20 zones across six zone types.';

-- The users table deliberately exposes only identity and home zone. The latent
-- behavioural traits that drive the simulation (price sensitivity, ETA
-- sensitivity, activity level, segment label) stay in users.parquet and are
-- never loaded here, so any user segmentation has to be learned from observed
-- behaviour instead of reading the generator's answer key.
CREATE TABLE users (
    user_id      TEXT PRIMARY KEY,
    home_zone_id TEXT NOT NULL REFERENCES zones (zone_id),
    signup_date  DATE NOT NULL
);

COMMENT ON TABLE users IS
    'One row per rider. Latent simulation traits are intentionally excluded.';

CREATE TABLE zone_state (
    ts                  TIMESTAMP        NOT NULL,
    zone_id             TEXT             NOT NULL REFERENCES zones (zone_id),
    hour                SMALLINT         NOT NULL,
    day_of_week         SMALLINT         NOT NULL,
    is_weekend          BOOLEAN          NOT NULL,
    is_peak_hour        BOOLEAN          NOT NULL,
    curfew              BOOLEAN          NOT NULL,
    air_raid_alert      BOOLEAN          NOT NULL,
    weather             TEXT             NOT NULL,
    special_event       BOOLEAN          NOT NULL,
    traffic_index       DOUBLE PRECISION NOT NULL,
    demand_count        INTEGER          NOT NULL CHECK (demand_count >= 0),
    available_drivers   INTEGER          NOT NULL CHECK (available_drivers >= 0),
    demand_supply_ratio DOUBLE PRECISION NOT NULL,
    supply_gap          INTEGER          NOT NULL,
    surge_multiplier    DOUBLE PRECISION NOT NULL
                        CHECK (surge_multiplier BETWEEN 1.0 AND 2.5),
    average_eta_minutes DOUBLE PRECISION NOT NULL
                        CHECK (average_eta_minutes > 0),
    PRIMARY KEY (ts, zone_id)
);

COMMENT ON TABLE zone_state IS
    'Marketplace state per zone per 15-minute interval: demand, supply, surge, ETA. '
    'curfew marks 00:00-05:00 when civilian movement is prohibited; air_raid_alert '
    'marks city-wide alerts.';

CREATE TABLE rides (
    ride_id                    BIGINT PRIMARY KEY,
    user_id                    TEXT             NOT NULL REFERENCES users (user_id),
    calculated_at              TIMESTAMP        NOT NULL,
    origin_zone_id             TEXT             NOT NULL REFERENCES zones (zone_id),
    destination_zone_id        TEXT             NOT NULL REFERENCES zones (zone_id),
    distance_km                DOUBLE PRECISION NOT NULL CHECK (distance_km > 0),
    estimated_duration_minutes DOUBLE PRECISION NOT NULL,
    eta_minutes                DOUBLE PRECISION NOT NULL CHECK (eta_minutes > 0),
    demand_count               INTEGER          NOT NULL,
    available_drivers          INTEGER          NOT NULL,
    demand_supply_ratio        DOUBLE PRECISION NOT NULL,
    traffic_index              DOUBLE PRECISION NOT NULL,
    weather                    TEXT             NOT NULL,
    is_peak_hour               BOOLEAN          NOT NULL,
    is_weekend                 BOOLEAN          NOT NULL,
    curfew                     BOOLEAN          NOT NULL,
    air_raid_alert             BOOLEAN          NOT NULL,
    special_event              BOOLEAN          NOT NULL,
    base_price                 DOUBLE PRECISION NOT NULL,
    surge_multiplier           DOUBLE PRECISION NOT NULL,
    final_price                DOUBLE PRECISION NOT NULL,
    placed                     SMALLINT         NOT NULL CHECK (placed IN (0, 1)),
    accepted                   SMALLINT         NOT NULL CHECK (accepted IN (0, 1)),
    churned_to_competitor      SMALLINT         NOT NULL CHECK (churned_to_competitor IN (0, 1)),
    search_wait_minutes        DOUBLE PRECISION,
    final_status               TEXT             NOT NULL
        CHECK (final_status IN ('calculated', 'accepted', 'churned_to_competitor')),
    placed_at                  TIMESTAMP,
    accepted_at                TIMESTAMP,

    -- The funnel invariants, enforced by the database rather than assumed.
    CONSTRAINT accepted_implies_placed        CHECK (accepted <= placed),
    CONSTRAINT churn_implies_placed           CHECK (churned_to_competitor <= placed),
    CONSTRAINT accepted_xor_churn_when_placed CHECK (
        placed = 0
        OR (accepted + churned_to_competitor = 1)
    ),
    CONSTRAINT search_wait_present_iff_placed CHECK (
        (search_wait_minutes IS NOT NULL) = (placed = 1)
    ),
    CONSTRAINT surge_never_discounts          CHECK (final_price >= base_price - 0.01),
    CONSTRAINT placed_at_present_iff_placed   CHECK ((placed_at IS NOT NULL) = (placed = 1)),
    CONSTRAINT accepted_at_present_iff_accept CHECK ((accepted_at IS NOT NULL) = (accepted = 1)),
    CONSTRAINT funnel_timestamps_ordered      CHECK (
        (placed_at IS NULL OR placed_at >= calculated_at)
        AND (accepted_at IS NULL OR accepted_at >= placed_at)
    )
);

COMMENT ON TABLE rides IS
    'One row per price calculation. After place: either accepted or churned '
    'to a competitor when search exceeds rider patience (~2-5 min).';
