import os
import sys
import docling.document_converter
import pymupdf
import pymupdf.layout
import pymupdf4llm
import docling
import torch
from typing import BinaryIO
from sqlmodel.ext.asyncio.session import AsyncSession
from core_db.models.job import Job

from docling.datamodel import pipeline_options, pipeline_options_vlm_model
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





class PdfProcessor: 

    def __init__(self):
        self.enrichPages = set()

    async def layoutAnalyzer(self, file: BinaryIO, session: AsyncSession):
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

        async with session.begin():
            ...

    
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
    



