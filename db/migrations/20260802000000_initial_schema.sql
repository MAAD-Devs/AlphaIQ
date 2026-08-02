-- migrate:up

CREATE TABLE users (
  email      TEXT        PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE portfolios (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_email   TEXT        NOT NULL REFERENCES users(email) ON DELETE CASCADE,
  name         TEXT        NOT NULL,
  account_drag FLOAT       NOT NULL DEFAULT 0.0,
  is_active    BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE portfolio_assets (
  id           UUID  PRIMARY KEY DEFAULT gen_random_uuid(),
  portfolio_id UUID  NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  ticker       TEXT  NOT NULL,
  asset_class  TEXT,
  display_name TEXT,
  annual_drag  FLOAT NOT NULL DEFAULT 0.0,
  dollar_value FLOAT NOT NULL
);

-- migrate:down

DROP TABLE portfolio_assets;
DROP TABLE portfolios;
DROP TABLE users;
