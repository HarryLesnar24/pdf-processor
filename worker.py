import heapq
import sys
import os
import socket
import asyncio
import aiofiles
from pathlib import Path
from collections import deque
from core_db.schemas.job import JobPriorityEnum
from core_db.schemas.task import TaskStatusEnum
from core_db.models.job import Job
from core_db.models.book import Book
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio.session import AsyncSession
from tenacity import (
    retry,
    wait_exponential_jitter,
    stop_after_attempt,
    retry_if_result,
    retry_if_exception_type,
    RetryError,
    AsyncRetrying,
)
from extractor import EmbeddingService, PdfProcessor
from session import asyncSession


PRIORITY_ORDER = [JobPriorityEnum.urgent, JobPriorityEnum.high, JobPriorityEnum.low]


containerID = socket.gethostname()


@retry(
    retry=retry_if_exception_type(Exception),
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
                task_status = :task,
                locked_by = :worker_id,
                locked_at = NOW(),
                heartbeat_at = NOW()
            FROM selected_batch
            WHERE jobs.job_uid = selected_batch.job_uid
            RETURNING jobs.*;
""")
    stmt = (
        select(Job)
        .from_statement(query)
        .params(worker_id=worker_id, task=TaskStatusEnum.running)
    )
    result = await session.execute(stmt)
    jobs = result.scalars().all()
    return jobs


async def downloadFile(path: str):
    # temporary
    ...


async def worker(container_id: str):
    EmbeddingService.initialize()
    print("Initialized Embedder")
    waitingTime = 60
    localQueue = {}
    while True:
        try:
            async with asyncSession() as session:
                assignedJobs = await hydrator(session, container_id)
                print("Hydrated Job")
                await session.commit()

            if not assignedJobs:
                await asyncio.sleep(60)
                continue
            if waitingTime > 60:
                waitingTime = 60

        except RetryError:
            await asyncio.sleep(waitingTime)
            waitingTime *= 2
            continue
        book_ids = [job.book_uid for job in assignedJobs]

        async with asyncSession() as session:
            results = await session.execute(
                select(Book.uid, Book.filepath).where(Book.uid.in_(book_ids)) # type: ignore
            ) 
            bookFileMap = {uid:path for uid, path in results.all()}
            print("Mapped Book UID with Filepath")
        for job in assignedJobs:
            filepath = bookFileMap.get(job.book_uid)
            if not filepath:
                continue
            if job.user_uid in localQueue:
                localQueue[job.user_uid][job.priority].append(
                (job.job_type, filepath, job)
                )
            else:
                localQueue[job.user_uid] = {
                    JobPriorityEnum.urgent: deque(),
                    JobPriorityEnum.high: deque(),
                    JobPriorityEnum.low: deque()
                }
            
                localQueue[job.user_uid][job.priority].append(
                (job.job_type, filepath, job)
                )
        print("LocalQueue Job Assigned")
        
        userQueue = deque(localQueue.keys())

        pdfProcessor = PdfProcessor()

        while userQueue:
            userUid = userQueue.popleft()
            print(f"Popped User: {userUid}")
            for priority in PRIORITY_ORDER:
                queue = localQueue[userUid][priority]

                if queue:
                    typeOfJob, path, jobObj = queue.popleft()
                    print(f'Path {path}, {jobObj.job_uid}')
                    await pdfProcessor.processor(jobObj, path)
                    print("Completed Job")
                    break
            if any(localQueue[userUid].values()):
                userQueue.append(userUid)


if __name__=="__main__":
    asyncio.run(worker(containerID))
 