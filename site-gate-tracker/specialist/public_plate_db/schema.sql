PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  source_url TEXT NOT NULL,
  license TEXT,
  production_policy TEXT,
  retrieved_at TEXT
);

CREATE TABLE IF NOT EXISTS media (
  media_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL REFERENCES sources(source_id),
  source_page_url TEXT,
  image_url TEXT,
  local_path TEXT,
  jurisdiction TEXT,
  author TEXT,
  license TEXT,
  license_url TEXT,
  width INTEGER,
  height INTEGER,
  sha256 TEXT,
  perceptual_hash TEXT,
  captured_at TEXT,
  imported_at TEXT NOT NULL,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS plate_observations (
  observation_id TEXT PRIMARY KEY,
  media_id TEXT NOT NULL REFERENCES media(media_id),
  bbox_x1 REAL,
  bbox_y1 REAL,
  bbox_x2 REAL,
  bbox_y2 REAL,
  plate_text TEXT,
  text_status TEXT NOT NULL CHECK(text_status IN (
    'none','machine_read','title_inferred','human_verified','registry_verified','synthetic_exact'
  )),
  jurisdiction TEXT,
  detector_confidence REAL,
  ocr_confidence REAL,
  plate_style TEXT,
  vehicle_id TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS vehicles (
  vehicle_id TEXT PRIMARY KEY,
  vin TEXT,
  make TEXT,
  model TEXT,
  variant TEXT,
  series TEXT,
  build_year INTEGER,
  body_type TEXT,
  vehicle_class TEXT,
  colour TEXT,
  gvm_kg REAL,
  gcm_kg REAL,
  tare_kg REAL,
  source_of_truth TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS vehicle_plates (
  vehicle_id TEXT NOT NULL REFERENCES vehicles(vehicle_id),
  plate_text TEXT NOT NULL,
  jurisdiction TEXT,
  valid_from TEXT,
  valid_to TEXT,
  evidence_source TEXT,
  PRIMARY KEY(vehicle_id,plate_text,jurisdiction)
);

CREATE INDEX IF NOT EXISTS idx_plate_text ON plate_observations(plate_text);
CREATE INDEX IF NOT EXISTS idx_plate_jurisdiction ON plate_observations(jurisdiction);
CREATE INDEX IF NOT EXISTS idx_media_source ON media(source_id);
CREATE INDEX IF NOT EXISTS idx_vehicle_vin ON vehicles(vin);
