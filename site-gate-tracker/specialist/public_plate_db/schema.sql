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


CREATE TABLE IF NOT EXISTS vehicle_resolutions (
  resolution_id TEXT PRIMARY KEY,
  plate_text TEXT NOT NULL,
  jurisdiction TEXT NOT NULL,
  vin TEXT,
  vehicle_id TEXT REFERENCES vehicles(vehicle_id),
  resolution_status TEXT NOT NULL CHECK(resolution_status IN (
    'unresolved','provider_candidate','provider_verified','manual_verified','rejected'
  )),
  provider TEXT,
  provider_reference TEXT,
  confidence REAL,
  resolved_at TEXT,
  raw_response_json TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS rav_records (
  vin TEXT PRIMARY KEY,
  rav_date_of_entry TEXT,
  entry_pathway_subcategory TEXT,
  approval_number TEXT,
  approval_holder TEXT,
  vehicle_category_code TEXT,
  vehicle_make TEXT,
  vehicle_model TEXT,
  authorised_by_name TEXT,
  build_date TEXT,
  gvm_atm_kg REAL,
  gtm_kg REAL,
  tare_kg REAL,
  motive_power TEXT,
  power_kw REAL,
  gcm_kg REAL,
  seating_capacity INTEGER,
  nves_vehicle_type TEXT,
  co2_g_km REAL,
  mass_in_running_order_kg REAL,
  source_url TEXT,
  fetched_at TEXT,
  raw_fields_json TEXT
);

CREATE TABLE IF NOT EXISTS training_labels (
  training_label_id TEXT PRIMARY KEY,
  observation_id TEXT NOT NULL REFERENCES plate_observations(observation_id),
  vehicle_id TEXT REFERENCES vehicles(vehicle_id),
  vin TEXT,
  plate_text TEXT,
  jurisdiction TEXT,
  make TEXT,
  model TEXT,
  variant TEXT,
  series TEXT,
  body_type TEXT,
  vehicle_class TEXT,
  gvm_kg REAL,
  gcm_kg REAL,
  tare_kg REAL,
  label_quality TEXT NOT NULL CHECK(label_quality IN (
    'weak','plate_verified','vin_verified','registry_enriched'
  )),
  provenance_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resolution_plate_state ON vehicle_resolutions(plate_text,jurisdiction);
CREATE INDEX IF NOT EXISTS idx_resolution_vin ON vehicle_resolutions(vin);
CREATE INDEX IF NOT EXISTS idx_rav_make_model ON rav_records(vehicle_make,vehicle_model);
CREATE INDEX IF NOT EXISTS idx_training_vin ON training_labels(vin);


CREATE TABLE IF NOT EXISTS camera_frames (
  frame_id TEXT PRIMARY KEY,
  camera_id TEXT NOT NULL,
  camera_title TEXT,
  camera_view TEXT,
  camera_region TEXT,
  direction TEXT,
  longitude REAL,
  latitude REAL,
  image_url TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  http_status INTEGER,
  sha256 TEXT,
  perceptual_hash TEXT,
  width INTEGER,
  height INTEGER,
  vehicle_count INTEGER DEFAULT 0,
  error TEXT
);

CREATE TABLE IF NOT EXISTS street_vehicle_observations (
  street_observation_id TEXT PRIMARY KEY,
  frame_id TEXT NOT NULL REFERENCES camera_frames(frame_id),
  media_id TEXT REFERENCES media(media_id),
  detector_class TEXT,
  detector_confidence REAL,
  bbox_x1 REAL,
  bbox_y1 REAL,
  bbox_x2 REAL,
  bbox_y2 REAL,
  crop_sha256 TEXT,
  crop_perceptual_hash TEXT,
  crop_width INTEGER,
  crop_height INTEGER,
  bbox_area_fraction REAL,
  edge_contact INTEGER,
  quality_score REAL,
  training_eligible INTEGER DEFAULT 0,
  quality_flags_json TEXT,
  provisional_plate TEXT,
  provisional_plate_confidence REAL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS camera_yield (
  camera_id TEXT PRIMARY KEY,
  camera_title TEXT,
  camera_region TEXT,
  attempts INTEGER DEFAULT 0,
  live_frames INTEGER DEFAULT 0,
  detected_vehicles INTEGER DEFAULT 0,
  retained_crops INTEGER DEFAULT 0,
  training_eligible_crops INTEGER DEFAULT 0,
  plate_reads INTEGER DEFAULT 0,
  score REAL DEFAULT 0,
  updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_camera_frames_camera_time
  ON camera_frames(camera_id,fetched_at);
CREATE INDEX IF NOT EXISTS idx_street_obs_frame
  ON street_vehicle_observations(frame_id);
CREATE INDEX IF NOT EXISTS idx_street_obs_plate
  ON street_vehicle_observations(provisional_plate);

CREATE INDEX IF NOT EXISTS idx_camera_yield_score ON camera_yield(score DESC);


CREATE TABLE IF NOT EXISTS plate_attempts (
  attempt_id TEXT PRIMARY KEY,
  media_id TEXT NOT NULL REFERENCES media(media_id),
  attempted_at TEXT NOT NULL,
  detector_variants INTEGER DEFAULT 0,
  detector_boxes INTEGER DEFAULT 0,
  plausible_plate_boxes INTEGER DEFAULT 0,
  best_detector_confidence REAL,
  ocr_attempts INTEGER DEFAULT 0,
  ocr_nonempty INTEGER DEFAULT 0,
  accepted INTEGER DEFAULT 0,
  accepted_text TEXT,
  accepted_confidence REAL,
  accepted_branch TEXT,
  accepted_native_plate_width REAL,
  failure_stage TEXT,
  diagnostics_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_plate_attempt_media ON plate_attempts(media_id);
CREATE INDEX IF NOT EXISTS idx_plate_attempt_stage ON plate_attempts(failure_stage);
