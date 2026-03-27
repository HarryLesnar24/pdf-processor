import heapq
import sys
import os
import socket
import asyncio
import aiofiles
from pathlib import Path
from core_db.models.job import Job
from core_db.models.book import Book
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from tenacity import (
    retry,
    wait_exponential_jitter,
    stop_after_attempt,
    retry_if_result,
    retry_if_exception_type,
    RetryError,
    AsyncRetrying,
)


asyncEngine = create_async_engine("postgresql+asyncpg://postgres@127.0.0.1:5432/appdb")
asyncSession = async_sessionmaker(
    bind=asyncEngine, class_=AsyncSession, expire_on_commit=False
)
containerID = socket.gethostname()


@retry(
    retry=retry_if_result(lambda r: not r),
    wait=wait_exponential_jitter(initial=1, max=900),
    stop=stop_after_attempt(10),
)
async def hydrator(session: AsyncSession, worker_id: str):
    query = text("""
        WITH prioritized_jobs AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_uid 
                    ORDER BY 
                        CASE 
                            WHEN job_type = 'focus' THEN 0 
                            WHEN job_type = 'bootstrap' THEN 1 
                            ELSE 2 
                        END ASC,
                        created_at ASC
                ) as user_job_rank
            FROM jobs
            WHERE task_status = 'queued'
            AND next_run_at <= NOW()
            AND locked_by IS NULL
        ),
            selected_batch AS (
                
                SELECT job_uid
                FROM prioritized_jobs
                WHERE user_job_rank <= 2 
                ORDER BY 
                    CASE 
                        WHEN job_type = 'focus' THEN 0 
                        ELSE 1 
                    END ASC,
                    created_at ASC
                LIMIT 10
                FOR UPDATE SKIP LOCKED
            )
            UPDATE jobs
            SET 
                task_status = 'processing',
                locked_by = :worker_id,
                locked_at = NOW(),
                heartbeat_at = NOW()
            FROM selected_batch
            WHERE jobs.job_uid = selected_batch.job_uid
            RETURNING jobs.*;
""")
    stmt = select(Job).from_statement(query).params(worker_id=worker_id)
    result = await session.execute(stmt)
    jobs = result.scalars().all()
    return jobs

async def downloadFile(path: str):
    # temporary 
    ...


async def worker(session: AsyncSession, container_id: str):
    waitingTime = 60
    localQueue = {}
    while True:
        try:
            assignedJobs = await hydrator(session, container_id)
            if waitingTime > 60:
                waitingTime = 60
        except RetryError:
            await asyncio.sleep(waitingTime)
            waitingTime *= 2
            continue

        for job in assignedJobs:
            stmt = select(Book.filepath).where(job.book_uid == Book.uid)  # type: ignore
            result = await session.execute(stmt)
            row = result.first()
            filepath = row[0] if row else None
            if job.user_uid in localQueue:
                heapq.heappush(
                    localQueue[job.user_uid],
                    (job.priority, job.job_type, filepath, job),
                )
            else:
                localQueue[job.user_uid] = []
                heapq.heappush(
                    localQueue[job.user_uid],
                    (job.priority, job.job_type, filepath, job),
                )
