PRAGMA foreign_keys=ON;

-- Initial schema for the ``items`` table.  In addition to the text
-- ``content`` and vector ``embedding`` each record stores optional
-- metadata such as timestamps, entities and importance scores used by the
-- memory subsystem.

CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    embedding BLOB,
    timestamp REAL,
    modality TEXT,
    entities TEXT,
    topics TEXT,
    importance REAL,
    recall_score REAL,
    schema_id INTEGER,
    frozen INTEGER DEFAULT 0,
    source TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    content,
    content='items',
    content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS items_hnsw USING hnsw(
    embedding FLOAT16[384]
);

CREATE TABLE IF NOT EXISTS edges (
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    relation TEXT,
    PRIMARY KEY (source_id, target_id, relation),
    FOREIGN KEY (source_id) REFERENCES items(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES items(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS schemas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    definition TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, content) VALUES (new.id, new.content);
    INSERT INTO items_hnsw(rowid, embedding) VALUES (new.id, new.embedding);
END;

CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
    DELETE FROM items_fts WHERE rowid = old.id;
    DELETE FROM items_hnsw WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
    UPDATE items_fts SET content = new.content WHERE rowid = new.id;
    UPDATE items_hnsw SET embedding = new.embedding WHERE rowid = new.id;
END;
