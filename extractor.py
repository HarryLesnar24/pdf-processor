import asyncio
from multiprocessing import connection
import os
from quopri import encodestring
import sys
from pathlib import Path
import docling.document_converter
import pymupdf
from dotenv import load_dotenv
import pymupdf.layout
import pymupdf4llm
import docling
# import torch
from sqlalchemy import func
from typing import BinaryIO
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlmodel import text
from core_db.models.job import Job
from core_db.models.page import Page
from core_db.schemas.page import PageIndexEnum, PageStatusEnum

from docling.datamodel import pipeline_options, pipeline_options_vlm_model
from docling_core.types.io import DocumentStream
from docling_core.types.doc.document import TextItem, CodeItem, FormulaItem
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    CodeFormulaVlmOptions,
    TableStructureOptions,
    TableFormerMode,
    RapidOcrOptions,
    OcrOptions,
    TesseractOcrOptions,
    LayoutOptions

)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.backend.docling_parse_backend import (
    DoclingParseDocumentBackend,
    DoclingParsePageBackend,
)
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from collections import defaultdict
from worker import asyncEngine
import logging

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

load_dotenv()

GPU_LOCK = os.getenv('GPU_ID')


class PdfProcessor: 

    def __imageExtractor(self, filePath: Path, imgbbox: dict, pagesState: dict):
        doc = pymupdf.open(filePath)

        for page_no, bboxes in imgbbox.items():
             page = doc[page_no - 1]

             extractedPaths = []

             for idx, bbox in enumerate(bboxes):
                  rect = pymupdf.Rect(bbox.l, bbox.t, bbox.r, bbox.b)

                  pix = page.get_pixmap(clip=rect, dpi=300)

                  img_filename = f"{pagesState[page_no]['book_uid']}_page_{page_no}_img_{idx}.png"
                  img_filepath = Path("tmp") / img_filename

                  pix.save(str(img_filepath))

                  extractedPaths.append(str(img_filepath))

             pagesState[page_no]['img_path'] = extractedPaths

        doc.close()



    async def layoutAnalyzer(self, filePath: Path, job: Job, session: AsyncSession):
        layout_options = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
            do_table_structure=False,
            do_ocr=False,
            layout_batch_size=8
        )
        layout_analyser = DocumentConverter(

            format_options={
                InputFormat.PDF : PdfFormatOption(
                    pipeline_options=layout_options,
                    backend=DoclingParseDocumentBackend
                )
            }
        )

        enrichPages = set()
        analyzedPages = {}
        imgBbox = defaultdict(list)
        async with asyncEngine.connect() as lock_conn:
            await lock_conn.execute(text(f'SELECT pg_advisory_lock({GPU_LOCK})'))
            try:
                layout_analyser.initialize_pipeline(InputFormat.PDF)
                doc = layout_analyser.convert(filePath, page_range=(job.page_start + 1, job.page_end + 1))
                for item, _ in doc.document.iterate_items():
                    page_no = item.prov[0].page_no # type: ignore
                    label = item.label # type: ignore
                    if page_no not in analyzedPages:
                            analyzedPages[page_no] = {
                                "page_no": page_no,
                                "book_uid": job.book_uid,
                                "user_uid": job.user_uid,
                                "index": PageIndexEnum.analyzed,
                                "required_deep": False
                            }
                    if label == "picture":
                        imgBbox[page_no].append(item.prov[0].bbox) # type: ignore
                        analyzedPages[page_no]["required_deep"] = True
                        analyzedPages[page_no]["has_image"] = True
                    elif label == "table":
                        analyzedPages[page_no]["has_table"] = True
                        analyzedPages[page_no]["required_deep"] = True
                    elif label == "formula":
                        analyzedPages[page_no]["has_formula"] = True
                        analyzedPages[page_no]["required_deep"] = True
                    elif label == "code":
                        analyzedPages[page_no]["has_code"] = True

                    if analyzedPages[page_no]["required_deep"]:
                            enrichPages.add(page_no)
            
            finally:
                    try:
                        await lock_conn.execute(text(f"SELECT pg_advisory_unlock({GPU_LOCK})"))
                    except Exception as e:
                        pass
        
        if imgBbox:
             await asyncio.to_thread(self.__imageExtractor, filePath, imgBbox, analyzedPages)
             
             
    
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
                    "updated_at": func.now()
                }
        ).returning(Page)
        async with session.begin():
            if batchData:
                result = await session.execute(stmtHandleConflict, batchData)
                upserted_pages = result.scalars().all()
                return upserted_pages, enrichPages


            

    
    async def pageExtractor(self, session: AsyncSession):

        ...
    
    async def enrichedPageExtractor(self, session: AsyncSession):
        enrich_options = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
            do_table_structure=True,
            do_formula_enrichment=True,
            do_ocr=True,
            code_formula_options=CodeFormulaVlmOptions.from_preset('codeformulav2'),
            ocr_options=TesseractOcrOptions(lang=['eng', 'equ']),
            table_structure_options=TableStructureOptions(mode=TableFormerMode.ACCURATE, do_cell_matching=False)

        )

        enrich_converter = DocumentConverter(
            format_options={
                InputFormat.PDF : PdfFormatOption(
                    backend=DoclingParseDocumentBackend,
                    pipeline_options=enrich_options
                )
            }
        )
    
    async def chunker(self, session: AsyncSession):
        ...
    
    async def embedder(self, session: AsyncSession):
        ...

    async def processor(self, job: Job, filepath: str, session: AsyncSession):
        ...
    



