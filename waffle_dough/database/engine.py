from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .model import Base

engine = None
conn = None


def create_session(db_path: str = None) -> sessionmaker:
    """Create a session.

    Args:
        db_path (str, optional): Path to the database. Defaults to ":memory:".

    Returns:
        sessionmaker: sessionmaker.
    """
    global engine, conn
    engine = create_engine(
        "sqlite://" + ("/" + db_path if db_path is not None else ""),
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    conn = engine.connect()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
