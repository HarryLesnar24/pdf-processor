import io
import gc
import re
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
import docling
import torch
from torch.cpu import is_available
import torch.nn.functional as F
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlmodel import text
from core_db.models.job import Job
from core_db.models.page import Page
from core_db.models.chunk import Chunk
from core_db.schemas.page import PageIndexEnum, PageStatusEnum
from core_db.vector.db import batchUpsert
from transformers import AutoTokenizer
from docling.datamodel import pipeline_options, pipeline_options_vlm_model
from docling_core.types.io import DocumentStream
from docling_core.types.doc.document import (
    ContentLayer,
    TextItem,
    CodeItem,
    FormulaItem,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    CodeFormulaVlmOptions,
    TableStructureOptions,
    TableFormerMode,
    RapidOcrOptions,
    OcrOptions,
    TesseractOcrOptions,
    LayoutOptions,
)

from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    DoclingParsePageBackend,
)
from chonkie import (
    MarkdownChef,
    Pipeline,
    CodeChunker,
    TableChunker,
    RecursiveChunker,
    Document,
    MarkdownDocument,
    RecursiveRules,
)
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from collections import defaultdict, OrderedDict
from session import asyncEngine, asyncSession
from optimum.onnxruntime import ORTModelForFeatureExtraction
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import tesserocr
import aiofiles
import logging
import json
import uuid

load_dotenv()


# rapid_engine = RapidOCR()
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
VECTOR_COLLECTION = os.getenv("COLLECTION_NAME")
APP_NAME = os.getenv("NAMESPACE_APP")
assert QDRANT_HOST != None
assert QDRANT_PORT != None
assert APP_NAME != None
NAMESPACE_UID = uuid.uuid5(uuid.NAMESPACE_DNS, APP_NAME)
OVERLAP_CHUNK_LIMIT = 400
GPU_LOCK = os.getenv("GPU_ID")
SPLIT_MARKER = "<!-- PAGE BREAK -->"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
tokenizerModel = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
asynClient = AsyncQdrantClient(host=QDRANT_HOST, port=int(QDRANT_PORT))


class EmbeddingService:
    _tokenizer = None
    _model = None

    @classmethod
    def initialize(cls):
        if cls._model is None:
            cls._tokenizer = tokenizerModel

            cls._model = ORTModelForFeatureExtraction.from_pretrained(
                EMBEDDING_MODEL,
                subfolder="onnx",
                file_name="model.onnx",
                provider="CPUExecutionProvider",
                trust_remote_code=True,
            )

    @staticmethod
    def _run_inference(model, inputs):
        with torch.no_grad():
            return model(**inputs)

    @classmethod
    async def batchEmbedding(cls, chunks: List[str]) -> List[List[float]]:
        inputs = cls._tokenizer(
            chunks, padding=True, truncation=True, return_tensors="pt"
        )  # type: ignore

        outputs = await asyncio.to_thread(cls._run_inference, cls._model, inputs)

        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
        pooled_embeddings = last_hidden_state[
            torch.arange(last_hidden_state.size(0)), sequence_lengths
        ]
        pooled_embeddings = pooled_embeddings[:, :768]
        normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        return normalized_embeddings.tolist()


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
        self, filePath: Path, imgbbox: dict, pagesState: dict, table=False
    ):
        doc = pymupdf.open(filePath)

        for page_no, bboxes in imgbbox.items():
            page = doc[page_no - 1]

            pageHeight = page.rect.height
            extractedPaths = []

            for idx, bbox in enumerate(bboxes):
                TaborImg = "image"
                topNew = pageHeight - bbox.t
                bottomNew = pageHeight - bbox.b
                rect = pymupdf.Rect(bbox.l, topNew, bbox.r, bottomNew)
                rect.normalize()
                pix = page.get_pixmap(clip=rect, dpi=300)

                if table:
                    TaborImg = "table"

                img_filename = f"{pagesState[page_no]['book_uid']}_page_{page_no}_{TaborImg}_{idx}.png"
                img_filepath = Path("tmp") / img_filename

                pix.save(str(img_filepath))

                extractedPaths.append(str(img_filepath))

            pagesState[page_no]["img_path"] = extractedPaths

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

    def __codeChunker(self, codeBlock: Dict[str, Any]):
        chunker = CodeChunker(
            language=codeBlock["language"].lower(),
            tokenizer=tokenizerModel,
            chunk_size=800,
        )
        chunks = chunker.chunk(codeBlock["content"])
        return [chunk.text for chunk in chunks]

    def __recursiveChunker(self, textBlock: str):
        chunker = RecursiveChunker(
            tokenizer=tokenizerModel, chunk_size=800, rules=RecursiveRules()
        )
        chunks = chunker.chunk(textBlock)
        return [chunk.text for chunk in chunks]
    
    async def __rescueFailedLayouts(self, filePath: Path, failedPages: Set[int], analyzedPages: dict, imgBbox: defaultdict, tableBbox: defaultdict):

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        rescueOptions = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
            do_table_structure=False,
            do_ocr=False,
            layout_batch_size=1
        )

        rescueAnalyzer = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=rescueOptions,
                    backend=DoclingParseDocumentBackend
                )
            }
        )


        async with asyncEngine.execution_options(isolation_level="AUTOCOMMIT").connect() as lock_conn:
            await lock_conn.execute(text(f"SELECT pg_advisory_lock({GPU_LOCK})"))
            rescueAnalyzer.initialize_pipeline(InputFormat.PDF)
            try:
                for failedPage in sorted(failedPages):
                    try:
                        doc = await asyncio.to_thread(
                            rescueAnalyzer.convert,
                            filePath,
                            page_range=(failedPage, failedPage),
                        )
                        
                        if doc and doc.status != ConversionStatus.FAILURE:
                            await self.layoutMapper(doc, analyzedPages, imgBbox, tableBbox)
                            
                    except Exception as e:
                        print(f"CRITICAL: Page {failedPage} failed even in safe mode: {e}")
                    
            finally:
                try:
                    await lock_conn.execute(text(f"SELECT pg_advisory_unlock({GPU_LOCK})"))
                except Exception:
                    pass

        

    async def layoutMapper(
        self,
        doc,
        pagesMetaData: dict,
        imgBbox: defaultdict,
        tableBbox: defaultdict,
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
                pagesMetaData[page_no]["has_code"] = True
                pagesMetaData[page_no]["required_deep"] = True


    async def layoutAnalyzer(self, filePath: Path, job: Job, session: AsyncSession):
        layout_options = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
            do_table_structure=False,
            do_ocr=False,
            layout_batch_size=4,
        )
        layout_analyser = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=layout_options, backend=DoclingParseDocumentBackend
                )
            }
        )
        doc = None
        enrichPages = set()
        analyzedPages = {}
        imgBbox = defaultdict(list)
        tableBbox = defaultdict(list)
        async with asyncEngine.execution_options(
            isolation_level="AUTOCOMMIT"
        ).connect() as lock_conn:
            await lock_conn.execute(text(f"SELECT pg_advisory_lock({GPU_LOCK})"))
            try:
                layout_analyser.initialize_pipeline(InputFormat.PDF)

                doc = await asyncio.to_thread(
                    layout_analyser.convert,
                    filePath,
                    page_range=(job.page_start + 1, job.page_end + 1),
                )

                failedPages: Set[int] = set()

                if doc.status in (
                    ConversionStatus.PARTIAL_SUCCESS,
                    ConversionStatus.FAILURE,
                ):
                    for error in doc.errors:
                        match = re.search(
                            r"pages? \[?(\d+)\]?", error.error_message, re.IGNORECASE
                        )
                        if match:
                            pageNum = int(match.group(1))
                            failedPages.add(pageNum)
            finally:
                try:
                    await lock_conn.execute(
                        text(f"SELECT pg_advisory_unlock({GPU_LOCK})")
                    )
                except Exception as e:
                    pass

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
                await self.layoutMapper(doc, analyzedPages, imgBbox, tableBbox, failedPages)
                await self.__rescueFailedLayouts(filePath, failedPages, analyzedPages, imgBbox, tableBbox)
        else:
            if doc:
                await self.layoutMapper(doc, analyzedPages, imgBbox, tableBbox)

        for page_no in range(job.page_start + 1, job.page_end + 2):
            if analyzedPages[page_no]["required_deep"]:
                enrichPages.add(page_no)

        if imgBbox:
            await asyncio.to_thread(
                self.__imageExtractor, filePath, imgBbox, analyzedPages
            )
        if tableBbox:
            await asyncio.to_thread(
                self.__imageExtractor, filePath, tableBbox, analyzedPages, table=True
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
        ) # type: ignore
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
                    "code": [],
                }
            else:
                md[page["metadata"]["page_number"]] = {
                    "markdown": page["text"],
                    "code": [],
                }
        return md


    async def enrichedPageExtractor(self, selectedPages: List[int], filePath: Path):
        enrich_options = ThreadedPdfPipelineOptions(
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

        enrich_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=DoclingParseDocumentBackend, pipeline_options=enrich_options
                )
            }
        )

        md = OrderedDict()

        pageMapper = {i + 1: actualNo for i, actualNo in enumerate(selectedPages)}
        buff = await asyncio.to_thread(self.__enrichDocCreator, filePath, selectedPages)
        stream = DocumentStream(name="temp.pdf", stream=buff)
        async with asyncEngine.execution_options(
            isolation_level="AUTOCOMMIT"
        ).connect() as lock_conn:
            await lock_conn.execute(text(f"SELECT pg_advisory_lock({GPU_LOCK})"))
            enrich_converter.initialize_pipeline(InputFormat.PDF)
            try:
                doc = await asyncio.to_thread(enrich_converter.convert, stream)
            finally:
                await lock_conn.execute(text(f"SELECT pg_advisory_unlock({GPU_LOCK})"))

        codeMaps = defaultdict(list)

        for item, _ in doc.document.iterate_items():
            if isinstance(item, CodeItem):  # type: ignore
                page = item.prov[0].page_no
                codeMaps[pageMapper[page]].append(
                    {
                        "language": item.code_language or "txt",
                        "content": item.text or None,
                        "bbox": item.prov[0].bbox,
                    }
                )

        markdownFile = doc.document.export_to_markdown(
            page_break_placeholder=SPLIT_MARKER
        )
        pages = markdownFile.split(SPLIT_MARKER)

        for page_no in range(1, len(doc.pages) + 1):
            actualPageNo = pageMapper[page_no]
            md[actualPageNo] = {
                "markdown": pages[page_no - 1],
                "code": codeMaps.get(actualPageNo, []),
            }

        buff.close()
        return md


    async def chunker(
        self,
        session: AsyncSession,
        md: OrderedDict,
        job: Job,
        pages: Sequence[Page],
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

        assert VECTOR_COLLECTION != None
        codeBlocks = None
        pagesMap = {page.page_no: page for page in pages}
        chunksObj = defaultdict(list)
        localData = []
        rollingBuffer = rollingOverlap
        book_id = job.book_uid
        for page_no, pageData in md.items():
            currentPage = pagesMap.get(page_no)
            markdown = pageData["markdown"]
            if pageData["code"]:
                codeBlocks = pageData["code"]

            if not markdown.strip():
                continue

            if rollingBuffer:
                markdown = rollingBuffer + "\n\n" + markdown

            mdDoc = await pipeline.arun(markdown)

            if len(markdown) > OVERLAP_CHUNK_LIMIT:
                prefix = markdown[-OVERLAP_CHUNK_LIMIT:]
                index = prefix.find(" ")
                rollingBuffer = prefix[index + 1 :]
            else:
                rollingBuffer = markdown

            codeChunks = []

            if codeBlocks:
                for block in codeBlocks:
                    if block["language"] != "txt":
                        chunks = await asyncio.to_thread(self.__codeChunker, block)
                        codeChunks.extend(chunks)
                    else:
                        chunks = await asyncio.to_thread(
                            self.__recursiveChunker, block["content"]
                        )
                        codeChunks.extend(chunks)

            if isinstance(mdDoc, Document):
                chunks = [chunk.text for chunk in mdDoc.chunks]

                if codeChunks:
                    chunks.extend(codeChunks)

                if not chunks:
                    continue

                vectors = await EmbeddingService.batchEmbedding(chunks)
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
                chunksObj[page_no].extend(pgChunks)
        async with aiofiles.open("chunks-1706.json", "w", encoding="utf-8") as file:
            jsonData = json.dumps(localData, indent=4)
            await file.write(jsonData)
        await session.commit()
        return chunksObj

        # await session.execute(
        #         update(Page)
        #         .where(Page.page_no.in_(selectedPages)) # type: ignore
        #         .values(index=PageIndexEnum.deep)
        # )

        # await session.flush()


    async def embedder(self, session: AsyncSession, chunk: str):
        model = ...

    async def processor(self, job: Job, filepath: Path):
        initialOverlap = ""
        async with asyncSession() as session:
            pagesObj, enrichPageSelected = await self.layoutAnalyzer(
                filepath, job, session
            )
        if pagesObj and enrichPageSelected:
            normalPageSelected = []
            for num in range(job.page_start, job.page_end + 1):
                if num + 1 not in enrichPageSelected:
                    normalPageSelected.append(num)
            textMd = await self.pageExtractor(normalPageSelected, filepath)
            enrichMd = await self.enrichedPageExtractor(
                sorted(enrichPageSelected), filepath
            )
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
                    session, markdown, job, pagesObj, initialOverlap
                )


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
