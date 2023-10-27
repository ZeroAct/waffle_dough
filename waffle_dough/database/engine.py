from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .model import Base

engine = create_engine(
    f"sqlite://" + (f"/{Settings.DB_PATH}" if Settings.DB_PATH else ""),
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
