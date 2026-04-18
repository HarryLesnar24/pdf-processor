from core_db.models.job import Job
from pathlib import Path
from core_db.schemas.job import JobPriorityEnum, JobTypeEnum
from extractor import EmbeddingService, PdfProcessor
import asyncio
import uuid


async def test():
    EmbeddingService.initialize()
    job = Job(
        book_uid=uuid.uuid4(),
        user_uid=uuid.uuid4(),
        priority=JobPriorityEnum.urgent,
        job_type=JobTypeEnum.focus,
        page_start=0,
        page_end=8,
        dedupekey="test-key",
    )  # type: ignore
    processor = PdfProcessor()
    filepath = Path("books/1706.03762v7.pdf")
    await processor.processor(job, filepath)


if __name__ == "__main__":
    asyncio.run(test())
