import io
import gc
from io import BytesIO
import asyncio
import heapq
import os
from typing import List, Dict, Any, Set, Optional, Sequence
from pathlib import Path
import pymupdf
from dotenv import load_dotenv
import pymupdf.layout
import pymupdf4llm
from pymupdf4llm.ocr import rapidtess_api
import torch
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio.session import AsyncSession
from core_ml.embedder.model import EmbeddingService
from core_db.models.job import Job
from core_db.models.page import Page
from core_db.models.chunk import Chunk
from core_db.schemas.task import TaskStatusEnum
from core_db.schemas.page import PageIndexEnum, PageStatusEnum
from core_db.schemas.book import BookStatusModel
from core_db.vector.db import batchUpsert
from transformers import AutoTokenizer
from docling_core.types.io import DocumentStream
from docling_core.types.doc.document import CodeItem
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    TableStructureOptions,
    TableFormerMode,
    RapidOcrOptions,
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from chonkie import Pipeline, CodeChunker, RecursiveChunker, Document, RecursiveRules
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from collections import defaultdict, OrderedDict
from session import asyncEngine, asyncSession
from mypy_boto3_s3 import S3Client
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiofiles
import json
import uuid

load_dotenv()


# rapid_engine = RapidOCR()
S3_BUCKET = os.getenv("S3_BUCKET")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
VECTOR_COLLECTION = os.getenv("COLLECTION_NAME")
APP_NAME = os.getenv("NAMESPACE_APP")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
assert QDRANT_HOST != None
assert QDRANT_PORT != None
assert APP_NAME != None
assert EMBEDDING_MODEL != None
NAMESPACE_UID = uuid.uuid5(uuid.NAMESPACE_DNS, APP_NAME)
OVERLAP_CHUNK_LIMIT = 400
GPU_LOCK = os.getenv("GPU_ID")
SPLIT_MARKER = "<!-- PAGE BREAK -->"

asynClient = AsyncQdrantClient(host=QDRANT_HOST, port=int(QDRANT_PORT))
tokenizerModel = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
doclingCodeToChonkieCode = {"C++": 'cpp', 'C#': 'csharp', 'unknown': 'txt'}

# embedder = EmbeddingService()

class DoclingModel:
    _layoutAnalyser = None
    _enrichConverter = None

    @classmethod
    def layoutAnalyzer(cls):
        if cls._layoutAnalyser is None:
            layoutOptions = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
                do_table_structure=False,
                do_ocr=False,
                layout_batch_size=2
            )
            cls._layoutAnalyser = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=layoutOptions,
                        backend=PyPdfiumDocumentBackend
                    )
                }
            )
            cls._layoutAnalyser.initialize_pipeline(InputFormat.PDF)
        return cls._layoutAnalyser
    
    @classmethod
    def enrichConverter(cls):
        if cls._enrichConverter is None:
            enrichOptions = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
                do_table_structure=True,
                do_formula_enrichment=True,
                do_code_enrichment=True,
                do_ocr=True,
                code_formula_options=CodeFormulaVlmOptions.from_preset("codeformulav2"),
                ocr_options=RapidOcrOptions(),
                table_structure_options=TableStructureOptions(
                    mode=TableFormerMode.ACCURATE, do_cell_matching=False
                ),
            )
            cls._enrichConverter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        backend=PyPdfiumDocumentBackend, 
                        pipeline_options=enrichOptions
                    )
                }
            )
            cls._enrichConverter.initialize_pipeline(InputFormat.PDF)
        return cls._enrichConverter
    
    
class PdfProcessor:
    def __enrichDocCreator(self, filePath: Path, pages: List[int]) -> BytesIO:
        with pymupdf.open(filePath) as src, pymupdf.open() as doc:
            for page in pages:
                doc.insert_pdf(docsrc=src, from_page=page - 1, to_page=page - 1)

            buffer = io.BytesIO()
            doc.save(buffer, garbage=1)

            buffer.seek(0)
            return buffer

    def __imageExtractor(
        self, filePath: Path, imgbbox: dict, pagesState: dict, s3: S3Client, table=False, code=False
    ):
        assert S3_BUCKET != None
        doc = pymupdf.open(filePath)
        futureMetadata = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            for page_no, bboxes in imgbbox.items():
                if not table:
                    print(f"Image Extractor Page Number: {page_no}")
                else:
                    print(f"Table Extractor Page Number: {page_no}")
                page = doc[page_no - 1]

                pagesState[page_no].setdefault("img_path", [])
                pagesState[page_no].setdefault("table_img_path", [])
                pagesState[page_no].setdefault("code_img_path", [])
                pageHeight = page.rect.height

                for idx, bbox in enumerate(bboxes):
                    topNew = pageHeight - bbox.t
                    bottomNew = pageHeight - bbox.b
                    rect = pymupdf.Rect(bbox.l, topNew, bbox.r, bottomNew)
                    rect.normalize()
                    pix = page.get_pixmap(clip=rect, dpi=300)

                    if not table and not code:
                        TaborImgorCod = "image"
                    elif table:
                        TaborImgorCod = "table"
                    else:
                        TaborImgorCod = "code"
                    
                    userUid = pagesState[page_no]["user_uid"]
                    bookUid = pagesState[page_no]["book_uid"]

                    imgS3Key = (
                        f"{userUid}/books/{bookUid}/page/{page_no}/{TaborImgorCod}/{idx}.png"
                    )
                    imgBytes = pix.tobytes("png")

                    future = executor.submit(
                        s3.put_object,
                        Bucket=S3_BUCKET,
                        Key=imgS3Key,
                        Body=imgBytes,
                        ContentType="image/png",
                    )

                    futureMetadata[future] = {"pageNo": page_no, "s3Key": imgS3Key}
            for future in as_completed(futureMetadata):
                metaData = futureMetadata[future]
                try:
                    future.result()
                    pageNo = metaData["pageNo"]
                    if not table and not code:
                        pagesState[pageNo]["img_path"].append(metaData["s3Key"])
                    elif table:
                        pagesState[pageNo]["table_img_path"].append(metaData["s3Key"])
                    else:
                        pagesState[pageNo]["code_img_path"].append(metaData["s3Key"])
                except Exception as e:
                    print(
                        f"❌ Failed to upload {metaData['pageNo']}, {metaData['s3Key']}: {e}"
                    )

        doc.close()

    def __textExtractor(self, filePath: Path, pages: List[int]) -> List[Dict[str, Any]]:
        doc = pymupdf.open(filePath)
        markdownDict = pymupdf4llm.to_markdown(
            doc,
            ocr_function=rapidtess_api.exec_ocr,
            ocr_language="eng+equ",
            page_chunks=True,
            pages=pages,
        )
        doc.close()
        # assert isinstance(markdownDict, List)
        return markdownDict  # type: ignore

    # def __codeChunker(self, codeBlock: Dict[str, Any]):
    #     chunker = CodeChunker(
    #         language=codeBlock["language"].lower(),
    #         tokenizer=tokenizerModel,
    #         chunk_size=800,
    #     )
    #     chunks = chunker.chunk(codeBlock["content"])
    #     return [chunk.text for chunk in chunks]

    # def __recursiveChunker(self, textBlock: str):
    #     chunker = RecursiveChunker(
    #         tokenizer=tokenizerModel, chunk_size=800, rules=RecursiveRules()
    #     )
    #     chunks = chunker.chunk(textBlock)
    #     return [chunk.text for chunk in chunks]

    async def __rescueFailedLayouts(
        self,
        filePath: Path,
        failedPages: Set[int],
        analyzedPages: dict,
        imgBbox: defaultdict,
        tableBbox: defaultdict,
        codeBbox: defaultdict
    ):

        # gc.collect()


        print(f"Rescue Layout Analyzer Failed Pages {failedPages}")

        # rescueOptions = ThreadedPdfPipelineOptions(
        #     accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
        #     do_table_structure=False,
        #     do_ocr=False,
        #     layout_batch_size=1,
        # )

        # rescueAnalyzer = DocumentConverter(
        #     format_options={
        #         InputFormat.PDF: PdfFormatOption(
        #             pipeline_options=rescueOptions, backend=DoclingParseDocumentBackend
        #         )
        #     }
        # )
        rescueAnalyzer = DoclingModel.layoutAnalyzer()
        async with asyncEngine.execution_options(
            isolation_level="AUTOCOMMIT"
        ).connect() as lock_conn:
            await lock_conn.execute(text(f"SELECT pg_advisory_lock({GPU_LOCK})"))
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # rescueAnalyzer.initialize_pipeline(InputFormat.PDF)
            try:
                for failedPage in sorted(failedPages):
                    doc = None
                    try:
                        doc = await asyncio.to_thread(
                            rescueAnalyzer.convert,
                            filePath,
                            page_range=(failedPage, failedPage),
                        )

                        for i, _ in doc.document.iterate_items():
                            print(f"Rescued Page Number: {i.prov[0].page_no}")  # type: ignore
                            break

                        await self.layoutMapper(doc, analyzedPages, imgBbox, tableBbox, codeBbox)
                    finally:
                        if doc:
                            del doc
                            gc.collect()
                            torch.cuda.empty_cache()

            finally:
                try:
                    await lock_conn.execute(
                        text(f"SELECT pg_advisory_unlock({GPU_LOCK})")
                    )

                except Exception:
                    pass

    async def __rescueFailedEnrichPages(
        self,
        # codeMapper: defaultdict,
        markdown: OrderedDict,
        failedPages: set,
        filePath: Path,
    ):
        # rescueEnrichOptions = ThreadedPdfPipelineOptions(
        #     accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
        #     do_table_structure=True,
        #     do_formula_enrichment=True,
        #     do_code_enrichment=True,
        #     do_ocr=False,
        #     code_formula_options=CodeFormulaVlmOptions.from_preset("codeformulav2"),
        #     ocr_options=RapidOcrOptions(),
        #     table_structure_options=TableStructureOptions(
        #         mode=TableFormerMode.ACCURATE, do_cell_matching=False
        #     ),
        # )

        # rescueEnrichConverter = DocumentConverter(
        #     format_options={
        #         InputFormat.PDF: PdfFormatOption(
        #             backend=DoclingParseDocumentBackend,
        #             pipeline_options=rescueEnrichOptions,
        #         )
        #     }
        # )

        # gc.collect()

        rescueEnrichConverter = DoclingModel.enrichConverter()

        print(f"Rescue Enrich Converter Failed Pages {failedPages}")

        async with asyncEngine.execution_options(
            isolation_level="AUTOCOMMIT"
        ).connect() as lock_conn:
            await lock_conn.execute(text(f"SELECT pg_advisory_lock({GPU_LOCK})"))
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # rescueEnrichConverter.initialize_pipeline(InputFormat.PDF)
            try:
                for failedPage in sorted(failedPages):
                    doc = None
                    try:
                        doc = await asyncio.to_thread(
                            rescueEnrichConverter.convert,
                            filePath,
                            page_range=(failedPage, failedPage),
                        )

                        # for item, _ in doc.document.iterate_items():
                        #     if isinstance(item, CodeItem):  # type: ignore
                        #         page = item.prov[0].page_no
                        #         codeMapper[page].append(
                        #             {
                        #                 "language": item.code_language.lower() if item.code_language not in doclingCodeToChonkieCode else doclingCodeToChonkieCode[item.code_language],
                        #                 "content": item.text or None,
                        #                 "bbox": item.prov[0].bbox,
                        #             }
                        #         )
                        markdown[failedPage] = {
                            "markdown": doc.document.export_to_markdown(),
                            # "code": codeMapper.get(failedPage, []),
                            "enriched": True,
                        }
                    finally:
                        if doc:
                            del doc
                            gc.collect()
                            torch.cuda.empty_cache()

            finally:
                try:
                    await lock_conn.execute(
                        text(f"SELECT pg_advisory_unlock({GPU_LOCK})")
                    )
                except Exception:
                    pass

    async def layoutMapper(
        self,
        doc,
        pagesMetaData: dict,
        imgBbox: defaultdict,
        tableBbox: defaultdict,
        codeBbox: defaultdict,
        failedPages: Optional[Set[int]] = None,
    ):
        for item, _ in doc.document.iterate_items():
            page_no = item.prov[0].page_no  # type: ignore

            if failedPages and page_no in failedPages:
                continue
            label = item.label  # type: ignore
            if label == "picture":
                imgBbox[page_no].append(item.prov[0].bbox)  # type: ignore
                pagesMetaData[page_no]["required_deep"] = True
                pagesMetaData[page_no]["has_image"] = True
            elif label == "table":
                tableBbox[page_no].append(item.prov[0].bbox)  # type: ignore
                pagesMetaData[page_no]["has_table"] = True
                pagesMetaData[page_no]["required_deep"] = True
            elif label == "formula":
                pagesMetaData[page_no]["has_formula"] = True
                pagesMetaData[page_no]["required_deep"] = True
            elif label == "code":
                codeBbox[page_no].append(item.prov[0].bbox) # type: ignore
                pagesMetaData[page_no]["has_code"] = True
                pagesMetaData[page_no]["required_deep"] = True

    async def layoutAnalyzer(
        self, filePath: Path, job: Job, session: AsyncSession, s3: S3Client
    ):
        # layout_options = ThreadedPdfPipelineOptions(
        #     accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
        #     do_table_structure=False,
        #     do_ocr=False,
        #     layout_batch_size=2,
        # )
        # layout_analyser = DocumentConverter(
        #     format_options={
        #         InputFormat.PDF: PdfFormatOption(
        #             pipeline_options=layout_options, backend=DoclingParseDocumentBackend
        #         )
        #     }
        # )

        # gc.collect()

        layout_analyser = DoclingModel.layoutAnalyzer()
        doc = None
        enrichPages = set()
        analyzedPages = {}
        imgBbox = defaultdict(list)
        tableBbox = defaultdict(list)
        codeBbox = defaultdict(list)
        async with asyncEngine.execution_options(
            isolation_level="AUTOCOMMIT"
        ).connect() as lock_conn:
            await lock_conn.execute(text(f"SELECT pg_advisory_lock({GPU_LOCK})"))
            try:
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()
                # layoutAnalyser.initialize_pipeline(InputFormat.PDF)

                doc = await asyncio.to_thread(
                    layout_analyser.convert,
                    filePath,
                    page_range=(job.page_start + 1, job.page_end + 1),
                )

            finally:
                try:
                    await lock_conn.execute(
                        text(f"SELECT pg_advisory_unlock({GPU_LOCK})")
                    )
                except Exception as e:
                    pass

            failedPages: Set[int] = set()
            successfulPages = set()
            requestedPages = set(range(job.page_start + 1, job.page_end + 2))

            if doc and hasattr(doc, "document") and doc.document:
                for item, _ in doc.document.iterate_items():
                    if hasattr(item, "prov") and item.prov:  # type: ignore
                        successfulPages.add(item.prov[0].page_no)  # type: ignore

            failedPages = requestedPages - successfulPages

        for page_no in range(job.page_start + 1, job.page_end + 2):
            if page_no not in analyzedPages:
                analyzedPages[page_no] = {
                    "page_no": page_no,
                    "book_uid": job.book_uid,
                    "user_uid": job.user_uid,
                    "index": PageIndexEnum.analyzed,
                    "required_deep": False,
                    "has_image": False,
                    "has_table": False,
                    "has_formula": False,
                    "has_code": False,
                }

        if failedPages:
            if doc:
                await self.layoutMapper(
                    doc, analyzedPages, imgBbox, tableBbox, codeBbox, failedPages
                )
                await self.__rescueFailedLayouts(
                    filePath, failedPages, analyzedPages, imgBbox, tableBbox, codeBbox
                )
        else:
            if doc:
                await self.layoutMapper(doc, analyzedPages, imgBbox, tableBbox, codeBbox)
        
        del doc
        gc.collect()
        torch.cuda.empty_cache()

        for page_no in range(job.page_start + 1, job.page_end + 2):
            if analyzedPages[page_no]["required_deep"]:
                enrichPages.add(page_no)

        if imgBbox:
            # with open("img.json", "w", encoding="utf-8") as f:
            #     serializable = {
            #         k: [bbox.__dict__ for bbox in v] for k, v in imgBbox.items()
            #     }
            #     json.dump(serializable, f, indent=4)
            await asyncio.to_thread(
                self.__imageExtractor, filePath, imgBbox, analyzedPages, s3
            )
        if tableBbox:
            # with open("table.json", "w", encoding="utf-8") as f:
            #     serializable = {
            #         k: [bbox.__dict__ for bbox in v] for k, v in tableBbox.items()
            #     }
            #     json.dump(serializable, f, indent=4)
            await asyncio.to_thread(
                self.__imageExtractor,
                filePath,
                tableBbox,
                analyzedPages,
                s3,
                table=True,
            )
        
        if codeBbox:
            await asyncio.to_thread(
                self.__imageExtractor,
                filePath,
                codeBbox,
                analyzedPages,
                s3,
                code=True
            )

        batchData = []
        for data in analyzedPages.values():
            newPage = Page(**data)
            batchData.append(newPage.model_dump())

        stmt = pg_insert(Page)
        stmtHandleConflict = stmt.on_conflict_do_update(
            constraint="uq_book_page",
            set_={
                "index": stmt.excluded.index,
                "required_deep": stmt.excluded.required_deep,
                "has_image": stmt.excluded.has_image,
                "has_table": stmt.excluded.has_table,
                "has_code": stmt.excluded.has_code,
                "has_formula": stmt.excluded.has_formula,
                "status": stmt.excluded.status,
                "img_path": stmt.excluded.img_path,
                "table_img_path": stmt.excluded.table_img_path,
                "code_img_path": stmt.excluded.code_img_path,
                "updated_at": func.now(),
            },
        ).returning(Page)
        if batchData:
            result = await session.execute(stmtHandleConflict, batchData)
            await session.commit()
            upserted_pages = result.scalars().all()
            return upserted_pages, enrichPages
        return [], []

    async def getPrevPageChunk(
        self, prevPageNo: int, bookUid: str, filepath: Path, session: AsyncSession
    ):
        stmt = (
            select(Chunk.chunk_data)
            .where((Chunk.book_uid == bookUid) & (Chunk.page_no == prevPageNo))
            .order_by(Chunk.chunk_index.desc())
            .limit(1)
        )  # type: ignore
        res = await session.execute(stmt)
        rawText = res.scalar_one_or_none()
        if not rawText:
            prevPageData = await self.pageExtractor([prevPageNo - 1], filepath)
            if not prevPageData:
                return ""
            rawText = list(prevPageData.values())[0]["markdown"]
        if len(rawText) > OVERLAP_CHUNK_LIMIT:
            prefix = rawText[-OVERLAP_CHUNK_LIMIT:]
            index = prefix.find(" ")
            return prefix[index + 1 :] if index != -1 else prefix
        return rawText

    async def pageExtractor(self, pages: List[int], filePath: Path):
        md = OrderedDict()
        markdownDicts: List[Dict[str, Any]] = await asyncio.to_thread(
            self.__textExtractor, filePath, pages
        )
        for page in markdownDicts:
            print(f"PyMuPdf Page No: {page['metadata']['page_number']}")
            if not page["text"].strip():
                continue
            removable = []
            for box in page["page_boxes"]:
                if (
                    box["class"] == "picture"
                    or box["class"] == "table"
                    or box["class"] == "formula"
                ):
                    removable.append(box["pos"])

            if removable:
                result = []
                lastIndex = 0
                for start, end in removable:
                    result.append(page["text"][lastIndex:start])
                    lastIndex = end
                result.append(page["text"][lastIndex:])
                md[page["metadata"]["page_number"]] = {
                    "markdown": "".join(result),
                    # "code": [],
                    "enriched": False,
                }
            else:
                md[page["metadata"]["page_number"]] = {
                    "markdown": page["text"],
                    # "code": [],
                    "enriched": False,
                }
        return md

    async def enrichedPageExtractor(self, selectedPages: List[int], filePath: Path):
        # enrich_options = ThreadedPdfPipelineOptions(
        #     accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
        #     do_table_structure=True,
        #     do_formula_enrichment=True,
        #     do_code_enrichment=True,
        #     do_ocr=False,
        #     code_formula_options=CodeFormulaVlmOptions.from_preset("codeformulav2"),
        #     ocr_options=RapidOcrOptions(),
        #     table_structure_options=TableStructureOptions(
        #         mode=TableFormerMode.ACCURATE, do_cell_matching=False
        #     ),
        # )

        # enrich_converter = DocumentConverter(
        #     format_options={
        #         InputFormat.PDF: PdfFormatOption(
        #             backend=DoclingParseDocumentBackend, pipeline_options=enrich_options
        #         )
        #     }
        # )

        md = OrderedDict()

        # gc.collect()

        enrich_converter = DoclingModel.enrichConverter()

        pageMapper = {i + 1: actualNo for i, actualNo in enumerate(selectedPages)}
        buff = await asyncio.to_thread(self.__enrichDocCreator, filePath, selectedPages)
        stream = DocumentStream(name="temp.pdf", stream=buff)
        async with asyncEngine.execution_options(
            isolation_level="AUTOCOMMIT"
        ).connect() as lock_conn:
            await lock_conn.execute(text(f"SELECT pg_advisory_lock({GPU_LOCK})"))
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # enrich_converter.initialize_pipeline(InputFormat.PDF)
            try:
                doc = await asyncio.to_thread(enrich_converter.convert, stream)
            finally:
                await lock_conn.execute(text(f"SELECT pg_advisory_unlock({GPU_LOCK})"))
                

        # codeMaps = defaultdict(list)
        failedPages = set()
        requestedPages = set(selectedPages)
        successfullPages = set()

        for item, _ in doc.document.iterate_items():
            page = pageMapper[item.prov[0].page_no]  # type: ignore
            successfullPages.add(page)
            # if isinstance(item, CodeItem):  # type: ignore
            #     page = item.prov[0].page_no
                
            #     codeMaps[pageMapper[page]].append(

            #         {
            #             "language": item.code_language.lower() if item.code_language not in doclingCodeToChonkieCode else doclingCodeToChonkieCode[item.code_language],
            #             "content": item.text or None,
            #             "bbox": item.prov[0].bbox,
            #         }
                
            #     )

        failedPages = requestedPages - successfullPages
        print(f"Failed Enrich Pages: {failedPages}")

        markdownFile = doc.document.export_to_markdown(
            page_break_placeholder=SPLIT_MARKER
        )
        pages = markdownFile.split(SPLIT_MARKER)

        for page_no in range(1, len(doc.pages) + 1):
            actualPageNo = pageMapper[page_no]
            print(f"Enriched Pages By Docling: {actualPageNo}")
            md[actualPageNo] = {
                "markdown": pages[page_no - 1],
                # "code": codeMaps.get(actualPageNo, []),
                "enriched": True,
            }

        buff.close()
        del doc
        gc.collect()
        torch.cuda.empty_cache()
        if failedPages:
            await self.__rescueFailedEnrichPages(md, failedPages, filePath)

        return md

    async def chunker(
        self,
        session: AsyncSession,
        md: OrderedDict,
        job: Job,
        pages: Sequence[Page],
        embedder: EmbeddingService,
        rollingOverlap: str = "",
    ):

        pipeline = (
            Pipeline()
            .process_with("markdown", tokenizer=tokenizerModel)
            .chunk_with(
                "recursive",
                chunk_size=800,
                tokenizer=tokenizerModel,
                rules=RecursiveRules(),
            )
            .refine_with(
                "overlap", tokenizer=tokenizerModel, context_size=100, method="prefix"
            )
        )

        gc.collect()

        assert VECTOR_COLLECTION != None
        assert EMBEDDING_MODEL != None
        pagesMap = {page.page_no: page for page in pages}
        chunksObj = defaultdict(list)
        localData = []
        rollingBuffer = rollingOverlap
        book_id = job.book_uid
        for page_no, pageData in md.items():
            currentPage = pagesMap.get(page_no)
            assert currentPage != None
            print(f"Chonkie Chunking the Page: {currentPage}")
            markdown = pageData["markdown"]
            # codeBlocks = pageData["code"]

            if not markdown.strip():
                continue

            if rollingBuffer:
                markdown = rollingBuffer + "\n\n" + markdown

            mdDoc = await pipeline.arun(markdown)

            currentPage.index = (
                PageIndexEnum.deep if pageData["enriched"] else PageIndexEnum.text
            )

            if len(markdown) > OVERLAP_CHUNK_LIMIT:
                prefix = markdown[-OVERLAP_CHUNK_LIMIT:]
                index = prefix.find(" ")
                rollingBuffer = prefix[index + 1 :]
            else:
                rollingBuffer = markdown

            # codeChunks = []

            # if codeBlocks:
            #     for block in codeBlocks:
            #         content = block.get("content")
            #         language = block.get("language", "txt")

            #         if not content or not content.strip():
            #             continue

            #         if language != "txt":
            #             chunks = await asyncio.to_thread(self.__codeChunker, block)
            #         else:
            #             chunks = await asyncio.to_thread(self.__recursiveChunker, content)

            #         codeChunks.extend(chunks)

            if isinstance(mdDoc, Document):
                chunks = [chunk.text for chunk in mdDoc.chunks]

                # if codeChunks:
                #     chunks.extend(codeChunks)

                if not chunks:
                    continue

                vectors = await embedder.batchEmbedding(chunks)
                stmt = select(Page.uid).where(
                    (Page.page_no == page_no) & (Page.book_uid == job.book_uid)
                )  # type: ignore
                result = await session.execute(stmt)
                page_uid = result.scalar_one_or_none()
                pgChunks = []
                payloads = []
                identifiers = []
                for index, chunk in enumerate(chunks):
                    data = Chunk(
                        page_uid=page_uid,
                        book_uid=job.book_uid,
                        user_uid=job.user_uid,
                        page_no=page_no,
                        chunk_index=index,
                        chunk_data=chunk,
                    )  # type: ignore
                    pgChunks.append(data)
                    localData.append(
                        {
                            "Chunk:": chunk,
                            "Chunk Index:": index,
                            "Page No:": page_no,
                        }
                    )
                session.add_all(pgChunks)
                await session.flush()

                for index in range(len(pgChunks)):
                    chunk = pgChunks[index]
                    payload = {
                        "user_uid": chunk.user_uid,
                        "book_uid": chunk.book_uid,
                        "page_uid": chunk.page_uid,
                        "chunk_uid": chunk.chunk_id,
                        "page_no": page_no,
                        "chunk_index": index,
                        "has_image": currentPage.has_image,  # type: ignore
                        "has_table": currentPage.has_table,  # type: ignore
                        "has_code": currentPage.has_code, # type: ignore
                        "img_s3keys": currentPage.img_path
                        if currentPage.has_image
                        else None,  # type: ignore
                        "table_s3keys": currentPage.table_img_path
                        if currentPage.has_table
                        else None,  # type: ignore
                        "code_s3keys": currentPage.code_img_path
                        if currentPage.has_code 
                        else None
                    }
                    payloads.append(payload)
                    template = f"{book_id}_page_{page_no}_chunk_{index}"
                    identity = uuid.uuid5(NAMESPACE_UID, template)
                    identifiers.append(identity)

                await batchUpsert(
                    asynClient,
                    collection=VECTOR_COLLECTION,
                    identifiers=identifiers,
                    payloads=payloads,
                    vectors=vectors,
                )
                currentPage.status = PageStatusEnum.complete
                chunksObj[page_no].extend(pgChunks)
                await session.merge(currentPage)
        async with aiofiles.open("chunks-C#.json", "a", encoding="utf-8") as file:
            jsonData = json.dumps(localData, indent=4)
            await file.write(jsonData)
        await session.merge(job)
        job.task_status = TaskStatusEnum.done
        await session.flush()
        stmt = text("""
            UPDATE books
            SET status = :state
            WHERE uid = :book_uid
            AND NOT EXISTS (
                    SELECT 1 FROM jobs
                    WHERE book_uid = :book_uid
                    AND task_status IN ('queued', 'running', 'failed', 'cancelled')
                    )
""")
        
        await session.execute(stmt, {"book_uid": job.book_uid, "state": BookStatusModel.ready_full})
        await session.commit()
        return chunksObj

        
    async def processor(self, job: Job, filepath: Path, s3: S3Client, embedder: EmbeddingService):
        try:
            print("=== JOB START ===")
            print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
            print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")
            assert EMBEDDING_MODEL != None
            print("Processor Started")
            initialOverlap = ""
            async with asyncSession() as session:
                pagesObj, enrichPageSelected = await self.layoutAnalyzer(
                    filepath, job, session, s3
                )
            print("=== After LayoutAnalyzer ===")
            print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
            print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")
            if pagesObj and enrichPageSelected:
                normalPageSelected = []
                for num in range(job.page_start, job.page_end + 1):
                    if num + 1 not in enrichPageSelected:
                        normalPageSelected.append(num)
                textMd = await self.pageExtractor(normalPageSelected, filepath)
                enrichMd = await self.enrichedPageExtractor(
                    sorted(enrichPageSelected), filepath
                )
                print("=== After EnrichedPageExtractor ===")
                print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
                print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")
                jobsPagesTotal = len(range(job.page_start + 1, job.page_end + 2))
                pagesProcessed = len(textMd) + len(enrichMd)
                # assert jobsPagesTotal == pagesProcessed
                print(f'JobsAssignedPagesTotal: {jobsPagesTotal}\nPagesProcessedTotal: {pagesProcessed}')
    
                mergeItems = heapq.merge(
                    textMd.items(), enrichMd.items(), key=lambda x: x[0]
                )
                markdown = OrderedDict(mergeItems)
            
                async with asyncSession() as session:
                    if job.page_start > 0:
                        initialOverlap = await self.getPrevPageChunk(
                            job.page_start, str(job.book_uid), filepath, session
                        )
                    chunksObj = await self.chunker(
                        session, markdown, job, pagesObj, embedder, initialOverlap
                    )
                print("=== After Chunker ===")
                print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
                print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")
        except Exception as e:
                    print(f"Job Failed {e}")
        finally:
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                print("=== JOB END ===")
                print(torch.cuda.memory_allocated() / 1024**2, "MB allocated")
                print(torch.cuda.memory_reserved() / 1024**2, "MB reserved")





    


# await session.execute(
        #         update(Page)
        #         .where(Page.page_no.in_(selectedPages)) # type: ignore
        #         .values(index=PageIndexEnum.deep)
        # )

        # await session.flush()



# def __ocr_handler(self, page: pymupdf.Page, **kwargs):
#     """
#     Custom OCR handler that replaces rapidtess_api.exec_ocr.
#     It intercepts the bounding boxes and clamps them to prevent Leptonica crashes.
#     """
#     # Get language from kwargs, defaulting to what you use in your textExtractor
#     lang = kwargs.get("language", "eng+equ")

#     # 1. Render the PDF page to a high-res image for OCR
#     dpi = 300
#     pix = page.get_pixmap(dpi=dpi)
#     img_width, img_height = pix.width, pix.height

#     # 2. Get bounding box coordinates from RapidOCR
#     img_bytes = pix.tobytes("png")
#     ocr_result, _ = rapid_engine(img_bytes)

#     if not ocr_result:
#         return # No text found on this page

#     # 3. Setup Tesserocr and loop through the boxes safely
#     with tesserocr.PyTessBaseAPI(lang=lang) as tess_api:
#         img_pil = Image.frombytes("RGB", [img_width, img_height], pix.samples) # type: ignore
#         tess_api.SetImage(img_pil)

#         for dt_box in ocr_result:
#             # dt_box format: [ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "text", confidence ]
#             box_coords = dt_box[0]

#             x_coords = [p[0] for p in box_coords]
#             y_coords = [p[1] for p in box_coords]

#             left = int(min(x_coords))
#             top = int(min(y_coords))
#             right = int(max(x_coords))
#             bottom = int(max(y_coords))

#             # --- THE FIX: CLAMP COORDINATES TO IMAGE BOUNDS ---
#             left = max(0, min(left, img_width - 1))
#             top = max(0, min(top, img_height - 1))
#             right = max(0, min(right, img_width))
#             bottom = max(0, min(bottom, img_height))

#             width = right - left
#             height = bottom - top

#             # Skip invalid or 0-pixel boxes
#             if width <= 0 or height <= 0:
#                 continue

#             # 4. Extract Text with Tesseract using the safe rectangle
#             tess_api.SetRectangle(left, top, width, height)
#             text = tess_api.GetUTF8Text().strip()

#             if text:
#                 # 5. Map coordinates back from the 300 DPI image to the PDF's point system (72 DPI)
#                 scale_x = page.rect.width / img_width
#                 scale_y = page.rect.height / img_height

#                 pdf_rect = pymupdf.Rect(
#                     left * scale_x,
#                     top * scale_y,
#                     right * scale_x,
#                     bottom * scale_y
#                 )

#                 # 6. Inject the text invisibly into the PDF page
#                 # render_mode=3 makes it an invisible, searchable text layer so pymupdf4llm can read it
#                 fontsize = max(4, pdf_rect.height * 0.8) # Approximate font size based on box height
#                 page.insert_text(
#                     pymupdf.Point(pdf_rect.x0, pdf_rect.y1), # Bottom-left of the bounding box
#                     text,
#                     fontsize=fontsize,
#                     render_mode=3
#                 )
