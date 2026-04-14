import io
from io import BytesIO
import asyncio
import heapq
import os
from typing import List, Dict, Any
from pathlib import Path
import pymupdf
from dotenv import load_dotenv
import pymupdf.layout
import pymupdf4llm
from pymupdf4llm.ocr import rapidtess_api
import docling
import torch
import torch.nn.functional as F
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlmodel import text
from core_db.models.job import Job
from core_db.models.page import Page
from core_db.models.chunk import Chunk
from core_db.schemas.page import PageIndexEnum, PageStatusEnum
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

from docling.datamodel.base_models import InputFormat
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

load_dotenv()

rapid_engine = RapidOCR()
GPU_LOCK = os.getenv("GPU_ID")
SPLIT_MARKER = "<!-- PAGE BREAK -->"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v5-text-small-retrieval"
tokenizerModel = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


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

        normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        return normalized_embeddings.tolist()


class PdfProcessor:

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
    
    def __enrichDocCreator(self, filePath: Path, pages: List[int]) -> BytesIO:
        with pymupdf.open(filePath) as src, pymupdf.open() as doc:
            for page in pages:
                doc.insert_pdf(docsrc=src, from_page=page - 1, to_page=page - 1)

            buffer = io.BytesIO()
            doc.save(buffer, garbage=1)

            buffer.seek(0)
            return buffer

    def __imageExtractor(self, filePath: Path, imgbbox: dict, pagesState: dict, table=False):
        doc = pymupdf.open(filePath)

        for page_no, bboxes in imgbbox.items():
            page = doc[page_no - 1]

            pageHeight = page.rect.height
            extractedPaths = []

            for idx, bbox in enumerate(bboxes):
                TaborImg = 'image'
                topNew = pageHeight - bbox.t
                bottomNew = pageHeight - bbox.b
                rect = pymupdf.Rect(bbox.l, topNew, bbox.r, bottomNew)
                rect.normalize()
                pix = page.get_pixmap(clip=rect, dpi=300)

                if table:
                    TaborImg = "table"

                img_filename = (
                    f"{pagesState[page_no]['book_uid']}_page_{page_no}_{TaborImg}_{idx}.png"
                )
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
        chunker = RecursiveChunker(tokenizer=tokenizerModel, chunk_size=800, )
        chunks = chunker.chunk(textBlock)
        return [chunk.text for chunk in chunks]

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
                
                for item, _ in doc.document.iterate_items():
                    page_no = item.prov[0].page_no  # type: ignore
                    label = item.label  # type: ignore
                    if label == "picture":
                        imgBbox[page_no].append(item.prov[0].bbox)  # type: ignore
                        analyzedPages[page_no]["required_deep"] = True
                        analyzedPages[page_no]["has_image"] = True
                    elif label == "table":
                        tableBbox[page_no].append(item.prov[0].bbox) # type: ignore
                        analyzedPages[page_no]["has_table"] = True
                        analyzedPages[page_no]["required_deep"] = True
                    elif label == "formula":
                        analyzedPages[page_no]["has_formula"] = True
                        analyzedPages[page_no]["required_deep"] = True
                    elif label == "code":
                        analyzedPages[page_no]["has_code"] = True
                        analyzedPages[page_no]["required_deep"] = True

                    if analyzedPages[page_no]["required_deep"]:
                        enrichPages.add(page_no)

            finally:
                try:
                    await lock_conn.execute(
                        text(f"SELECT pg_advisory_unlock({GPU_LOCK})")
                    )
                except Exception as e:
                    pass

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

    async def pageExtractor(
        self, pages: List[int], filePath: Path
    ):
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
                md[page["metadata"]["page_number"]] = {'markdown': "".join(result), 'code': []}
            else:
                md[page["metadata"]["page_number"]] = {'markdown': page["text"], 'code': []}
        return md

    async def enrichedPageExtractor(
        self, selectedPages: List[int], filePath: Path
    ):
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
        self, session: AsyncSession, md: OrderedDict, job: Job
    ):
        pipeline = (
            Pipeline()
            .process_with("markdown", tokenizer=tokenizerModel)
            .chunk_with(
                 "recursive",
                 chunk_size=800,
                 tokenizer=tokenizerModel,
                 rules=RecursiveRules()
            )
            .refine_with(
                "overlap", tokenizer=tokenizerModel, context_size=100, method="prefix"
            )
        )


        codeBlocks = None

        chunksObj = defaultdict(list)

        localData = []
        for page_no, pageData in md.items():
            markdown = pageData["markdown"]
            if pageData["code"]:
                codeBlocks = pageData["code"]
            if not markdown.strip():
                continue

            mdDoc = await pipeline.arun(markdown)

            codeChunks = []

            if codeBlocks:
                for block in codeBlocks:
                    if block["language"] != "txt":
                        chunks = await asyncio.to_thread(self.__codeChunker, block)
                        codeChunks.extend(chunks)
                    else:
                        chunks = await asyncio.to_thread(self.__recursiveChunker, block["content"])
                        codeChunks.extend(chunks)

            if isinstance(mdDoc, Document):
                chunks = [chunk.text for chunk in mdDoc.chunks]

                if codeChunks:
                    chunks.extend(codeChunks)
                vectors = await EmbeddingService.batchEmbedding(chunks)
                stmt = select(Page.uid).where(Page.page_no == page_no)  # type: ignore
                result = await session.execute(stmt)
                page_uid = result.scalar_one_or_none()
                pgChunks = []
                for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    data = Chunk(
                        page_uid=page_uid,
                        book_uid=job.book_uid,
                        user_uid=job.user_uid,
                        page_no=page_no,
                        chunk_index=index,
                        chunk_data=chunk
                    ) # type: ignore
                    pgChunks.append(data)
                    dictionary = {
                        'Chunk:': chunk,
                        'Vector:': vector
                    }
                    localData.append(dictionary)
                    
                stmt = pg_insert(Chunk).returning(Chunk)
                res = await session.execute(stmt, pgChunks)

                chunksObj[page_no].append(res.scalars().all())
        async with aiofiles.open('chunk-vector.json', 'w', encoding='utf-8') as file:
             jsonData = json.dumps(localData, indent= 4)
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
        async with asyncSession() as session:
            pagesObj, enrichPageSelected = await self.layoutAnalyzer(filepath, job, session)
        normalPageSelected = []
        for num in range(job.page_start, job.page_end + 1):
            if num + 1 not in enrichPageSelected:
                normalPageSelected.append(num)
        textMd = await self.pageExtractor(normalPageSelected, filepath)
        enrichMd = await self.enrichedPageExtractor(sorted(enrichPageSelected), filepath)
        mergeItems = heapq.merge(textMd.items(), enrichMd.items(), key=lambda x: x[0])
        markdown = OrderedDict(mergeItems)
        async with asyncSession() as session:
            chunksObj = await self.chunker(session, markdown, job)
        

        
        
        







# print(torch.__version__)


# pipeline_options = ThreadedPdfPipelineOptions(
#     accelerator_options=AcceleratorOptions(
#         device=AcceleratorDevice.CUDA, cuda_use_flash_attention2=False
#     ),
#     do_ocr=False,
#      table_structure_options=TableStructureOptions(
#          mode=TableFormerMode.ACCURATE, do_cell_matching=False
#     ),
#     code_formula_options=CodeFormulaVlmOptions.from_preset("codeformulav2"),
#     do_formula_enrichment=False,
#     do_code_enrichment=False,
#     do_table_structure=False,
#     ocr_batch_size=4,
#     layout_batch_size=4,
#     table_batch_size=4,
# )

# doc_converter = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(
#             backend=DoclingParseDocumentBackend,
#             pipeline_options=pipeline_options,
#         )
#     }
# )

# 3. Initialize the pipeline
# doc_converter.initialize_pipeline(InputFormat.PDF)


# 4. Convert your document

# with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
# result = doc_converter.convert(
#     "C Sharp in Depth.pdf", page_range=(75, 87)
# )

# with open('docling-preview/list-3.txt', 'w', encoding='utf-8') as f:
# f.write(str(result.document.texts))

# for item, level in result.document.iterate_items():

# if isinstance(item, TextItem):
#     print(item.get_ref())
#      print(f'Label {item.label} Prov: {item.prov[0].page_no}') # type: ignore
#     print(item.hyperlink)
#     print('*' * 45)

# print(f'{item.label} Page-No {item.prov[0].page_no}') # type: ignore
# Access the converted content
# with open("docling-preview/docling-test-8.md", "w", encoding="utf-8") as f:
#     f.write(result.document.export_to_markdown())


# from docling.document_converter import DocumentConverter
# from docling_core.types.doc import PictureItem, PictureMeta, DescriptionMetaField

# converter = DocumentConverter()
# result = converter.convert("document.pdf")
# doc = result.document

# # Iterate and add captions from your API
# for pic in doc.pictures:
#     img = pic.get_image(doc) # Returns PIL Image
#     if img is None:
#         continue

#     # Call your remote API
#     caption = call_your_api(img)

#     # Set the description
#     if pic.meta is None:
#         pic.meta = PictureMeta()
#     pic.meta.description = DescriptionMetaField(text=caption)

# def export_with_captions(doc):
#     md_lines = []
#     for item, level in doc.iterate_items():
#         if isinstance(item, PictureItem):
#             alt_text = "Image"
#             if item.meta and item.meta.description:
#                 alt_text = item.meta.description.text

#             if item.image and item.image.uri:
#                 md_lines.append(f"![{alt_text}]({item.image.uri})")
#             else:
#                 md_lines.append(f"<!-- {alt_text} -->")
#         # Handle other item types...
#     return "\n\n".join(md_lines)

# for item, level in conv_result.document.iterate_items():
#     if isinstance(item, CodeItem):
#         print(f"Language: {item.code_language}")
#         print(f"Code: {item.text}")
