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

def seed_default_data():
    """Seeds default curriculum chunks and questions if DB is empty."""
    from app.models import SourceDocuments, ContentChunks, Questions
    db = SessionLocal()
    try:
        if db.query(SourceDocuments).count() == 0:
            doc = SourceDocuments(
                id="SEED-DOC-001",
                file_name="CogniAdapt_Core_Curriculum.pdf",
                grade=10,
                subject="Science & AI"
            )
            db.add(doc)

            c1 = ContentChunks(
                id="SEED-CHUNK-001",
                source_id="SEED-DOC-001",
                chunk_text="Photosynthesis is the process used by plants, algae, and cyanobacteria to convert light energy into chemical energy stored in glucose. Chlorophyll pigment absorbs blue and red light.",
                topic="Science",
                subtopic="Biology",
                page_number=12
            )
            c2 = ContentChunks(
                id="SEED-CHUNK-002",
                source_id="SEED-DOC-001",
                chunk_text="Deep Q-Learning (DQN) combines deep neural networks with Q-learning to approximate optimal Q-values Q(s, a). Item Response Theory (IRT) estimates student ability theta and item difficulty b.",
                topic="Science",
                subtopic="AI & Reinforcement Learning",
                page_number=45
            )
            db.add_all([c1, c2])

            q1 = Questions(
                id="SEED-Q-001",
                source_chunk_id="SEED-CHUNK-001",
                question="Which pigment absorbs light energy in plant cells during photosynthesis?",
                type="MCQ",
                options='["Chlorophyll", "Carotenoid", "Anthocyanin", "Hemoglobin"]',
                answer="Chlorophyll",
                difficulty="easy",
                page_number=12
            )
            q2 = Questions(
                id="SEED-Q-002",
                source_chunk_id="SEED-CHUNK-001",
                question="What is the primary energy transformation in photosynthesis?",
                type="MCQ",
                options='["Light energy into chemical energy", "Chemical energy into heat", "Nuclear energy into light", "Kinetic energy into electricity"]',
                answer="Light energy into chemical energy",
                difficulty="medium",
                page_number=12
            )
            q3 = Questions(
                id="SEED-Q-003",
                source_chunk_id="SEED-CHUNK-002",
                question="In Reinforcement Learning, what does the Q-value Q(s, a) represent?",
                type="MCQ",
                options='["Expected cumulative reward for taking action a in state s", "Instant prediction error", "Number of neural network layers", "Learning rate multiplier"]',
                answer="Expected cumulative reward for taking action a in state s",
                difficulty="hard",
                page_number=45
            )
            q4 = Questions(
                id="SEED-Q-004",
                source_chunk_id="SEED-CHUNK-002",
                question="Which framework estimates student skill ability theta based on response accuracy?",
                type="MCQ",
                options='["Item Response Theory (IRT)", "Linear Regression", "K-Means Clustering", "Fourier Transform"]',
                answer="Item Response Theory (IRT)",
                difficulty="medium",
                page_number=45
            )
            q5 = Questions(
                id="SEED-Q-005",
                source_chunk_id="SEED-CHUNK-001",
                question="True or False: Oxygen is produced as a byproduct during photosynthesis.",
                type="True/False",
                options='["True", "False"]',
                answer="True",
                difficulty="easy",
                page_number=12
            )
            db.add_all([q1, q2, q3, q4, q5])
            db.commit()
            print("[DB Seed] Successfully seeded CogniAdapt default curriculum and questions.")
    except Exception as e:
        db.rollback()
        print(f"[DB Seed] Warning: {e}")
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

