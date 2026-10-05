import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./peblo.db")

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def auto_migrate_sqlite():
    """Ensures SQLite schema contains all new columns without data loss."""
    if "sqlite" not in DATABASE_URL:
        return
    with engine.connect() as conn:
        from sqlalchemy import text
        # ContentChunks columns
        existing_cc = [row[1] for row in conn.execute(text("PRAGMA table_info(content_chunks)")).fetchall()]
        for col, col_type in [("subtopic", "VARCHAR"), ("page_number", "INTEGER"), ("slide_number", "INTEGER"), ("video_timestamp", "VARCHAR")]:
            if col not in existing_cc and existing_cc:
                try:
                    conn.execute(text(f"ALTER TABLE content_chunks ADD COLUMN {col} {col_type}"))
                    print(f"[DB Migration] Added column '{col}' to content_chunks table.")
                except Exception as e:
                    pass

        # Questions columns
        existing_q = [row[1] for row in conn.execute(text("PRAGMA table_info(questions)")).fetchall()]
        for col, col_type in [("page_number", "INTEGER"), ("slide_number", "INTEGER"), ("video_timestamp", "VARCHAR")]:
            if col not in existing_q and existing_q:
                try:
                    conn.execute(text(f"ALTER TABLE questions ADD COLUMN {col} {col_type}"))
                    print(f"[DB Migration] Added column '{col}' to questions table.")
                except Exception as e:
                    pass
        conn.commit()

# Run migration
try:
    auto_migrate_sqlite()
except Exception as e:
    print(f"[DB Migration] Warning: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
