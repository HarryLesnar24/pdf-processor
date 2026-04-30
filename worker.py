import heapq
import sys
import os
import socket
import asyncio
import aiofiles
import boto3
from pathlib import Path
from collections import deque
from core_db.schemas.job import JobPriorityEnum
from core_db.schemas.task import TaskStatusEnum
from core_db.models.job import Job
from core_db.models.book import Book
from dotenv import load_dotenv
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio.session import AsyncSession
from tenacity import (
    retry,
    wait_exponential_jitter,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
)
from typing import cast
from extractor import PdfProcessor
from core_ml.embedder.model import EmbeddingService
from session import asyncSession
from mypy_boto3_s3 import S3Client
import faulthandler
import logging



faulthandler.enable(file=sys.stderr, all_threads=True)

load_dotenv()


logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s [%(levelname)s] pid:%(process)d %(name)s: %(message)s"
)

PRIORITY_ORDER = [JobPriorityEnum.urgent, JobPriorityEnum.high, JobPriorityEnum.low]

S3_BUCKET = os.getenv("S3_BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_ACCESS_SECRET_KEY = os.getenv("AWS_ACCESS_SECRET_KEY")
REGION = os.getenv("REGION")
DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", 120))


assert REGION != None
assert S3_BUCKET != None
assert S3_ENDPOINT != None
assert AWS_ACCESS_KEY_ID != None
assert AWS_ACCESS_SECRET_KEY != None
assert EMBEDDING_MODEL != None
embedder = EmbeddingService()
containerID = socket.gethostname()


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential_jitter(initial=1, max=900),
    stop=stop_after_attempt(10),
)
async def heartbeater(workerId: str, interval: int):
    while True:
        try:
            await asyncio.sleep(interval)
            async with asyncSession() as session:
                stmt = text("""
                            UPDATE jobs
                            SET heartbeat_at = NOW()
                            WHERE locked_by = :worker_id
                            AND task_status = :task
                            """)
                await session.execute(stmt, {"worker_id": workerId, "task": TaskStatusEnum.running})
                await session.commit()
        except asyncio.CancelledError:
            print("Heartbeater task cancelled. Shutting down heartbeat.")
            break
        except Exception as e:
            print(f"Failed to update heartbeat: {e}")


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


async def downloadFile(filename: str, s3Key: str, s3: S3Client, downloadFolder: Path):
    assert S3_BUCKET != None
    await asyncio.to_thread(
        s3.download_file,
        Bucket=S3_BUCKET,
        Key=s3Key,
        Filename=str(downloadFolder.absolute() / filename),
    )


async def checkFileExists(directory: Path, filename: str) -> bool:
    if not directory.is_dir():
        os.makedirs(directory, exist_ok=True)
    filePath = directory / filename
    if filePath.exists() and filePath.is_file():
        return True
    return False


async def worker(containerId: str):
    assert DOWNLOAD_FOLDER != None
    assert EMBEDDING_MODEL != None
    s3 = cast(
        S3Client,
        boto3.client(
            service_name="s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_ACCESS_SECRET_KEY,
            region_name=REGION,
        ),
    )
    downloadFolder = Path(DOWNLOAD_FOLDER)
    embedder.initialize(EMBEDDING_MODEL)
    print("Initialized Embedder")
    heartbeatTask = asyncio.create_task(
        heartbeater(containerId, HEARTBEAT_INTERVAL)
    )
    localQueue = {}
    try:
        while True:
            try:
                async with asyncSession() as session:
                    assignedJobs = await hydrator(session, containerId)

                    print("Hydrated Job")

                    await session.commit()

                if not assignedJobs:
                    await asyncio.sleep(60)
                    continue

            except RetryError:
                await asyncio.sleep(60)
                continue

            book_ids = [job.book_uid for job in assignedJobs]

            async with asyncSession() as session:
                results = await session.execute(
                    select(Book.uid, Book.filepath).where(Book.uid.in_(book_ids))  # type: ignore
                )
                bookFileMap = {uid: path for uid, path in results.all()}

                print("Mapped Book UID with Filepath")

            for job in assignedJobs:
                filepath = bookFileMap.get(job.book_uid)

                if not filepath:
                    continue
                filename = filepath.split("/")[-1]

                if not await checkFileExists(downloadFolder, filename):
                    await downloadFile(filename, filepath, s3, downloadFolder)
                    print(f"Downloaded to the tmp folder {filename}")

                filepath = downloadFolder.absolute() / filename

                if job.user_uid in localQueue:
                    localQueue[job.user_uid][job.priority].append(
                        (job.job_type, filepath, job)
                    )
                else:
                    localQueue[job.user_uid] = {
                        JobPriorityEnum.urgent: deque(),
                        JobPriorityEnum.high: deque(),
                        JobPriorityEnum.low: deque(),
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
                        print(f"Path {path}, {jobObj.job_uid}")
                        await pdfProcessor.processor(jobObj, path, s3, embedder)
                        print("Completed Job")
                        break

                if any(localQueue[userUid].values()):
                    userQueue.append(userUid)
    finally:
        heartbeatTask.cancel()

if __name__ == "__main__":
    asyncio.run(worker(containerID))
