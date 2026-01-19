-- CPU Index Builder - PostgreSQL Schema
-- This file is automatically run on container initialization

-- Raw API responses for audit trail
CREATE TABLE IF NOT EXISTS raw_responses (
    id SERIAL PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL,
    month VARCHAR(7) NOT NULL,
    response_json JSONB NOT NULL,
    fetched_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_raw_responses_month ON raw_responses(month);
CREATE INDEX IF NOT EXISTS idx_raw_responses_query_hash ON raw_responses(query_hash);

-- Parsed articles from LexisNexis
CREATE TABLE IF NOT EXISTS articles (
    id VARCHAR(255) PRIMARY KEY,  -- LexisNexis ResultId
    title TEXT,
    date DATE,
    source VARCHAR(255),
    snippet TEXT,
    month VARCHAR(7) NOT NULL,
    fetched_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_articles_month ON articles(month);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(date);

-- Local keyword classifications
CREATE TABLE IF NOT EXISTS keyword_classifications (
    article_id VARCHAR(255) PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    has_uncertainty BOOLEAN DEFAULT FALSE,
    has_reversal_terms BOOLEAN DEFAULT FALSE,
    has_implementation_terms BOOLEAN DEFAULT FALSE,
    has_upside_terms BOOLEAN DEFAULT FALSE,
    has_ira_mention BOOLEAN DEFAULT FALSE,
    has_obbba_mention BOOLEAN DEFAULT FALSE,
    matched_terms JSONB,  -- {"uncertainty": ["uncertain"], "reversal": ["rollback"]}
    classified_at TIMESTAMP DEFAULT NOW()
);

-- LLM validation results
CREATE TABLE IF NOT EXISTS llm_classifications (
    article_id VARCHAR(255) PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    is_climate_policy BOOLEAN,
    uncertainty_type VARCHAR(20),  -- 'implementation', 'reversal', 'none'
    certainty_level INTEGER CHECK (certainty_level >= 1 AND certainty_level <= 5),
    reasoning TEXT,
    model VARCHAR(50),
    classified_at TIMESTAMP DEFAULT NOW()
);

-- Computed indices (cached results)
CREATE TABLE IF NOT EXISTS index_values (
    month VARCHAR(7) NOT NULL,
    index_type VARCHAR(50) NOT NULL,  -- 'cpu', 'cpu_reversal', 'cpu_impl', 'regime_ira', etc.
    outlet VARCHAR(100) DEFAULT '',  -- Empty string for aggregate, outlet name for per-outlet
    denominator INTEGER NOT NULL,
    numerator INTEGER NOT NULL,
    raw_ratio REAL,
    normalized REAL,
    scaled BOOLEAN DEFAULT FALSE,
    dedup_strategy VARCHAR(50) DEFAULT 'none',
    computed_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (month, index_type, outlet, scaled, dedup_strategy)
);
CREATE INDEX IF NOT EXISTS idx_index_values_type ON index_values(index_type);

-- Collection progress tracking
CREATE TABLE IF NOT EXISTS collection_progress (
    month VARCHAR(7) PRIMARY KEY,
    articles_fetched INTEGER DEFAULT 0,
    completed_at TIMESTAMP
);

-- Grant permissions (for safety if running as different user)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cpu_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cpu_user;
