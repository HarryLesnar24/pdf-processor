from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.ext.asyncio.session import AsyncSession
import os
from dotenv import load_dotenv


load_dotenv()
asyncEngine = create_async_engine(
    f"postgresql+asyncpg://postgres:{os.getenv('DB_PASSWORD')}@127.0.0.1:5432/appdb"
)
asyncSession = async_sessionmaker(
    bind=asyncEngine, class_=AsyncSession, expire_on_commit=False
)
