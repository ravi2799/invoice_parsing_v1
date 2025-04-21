import uvicorn
import os
import io
import time
import logging
import concurrent.futures
from typing import List, Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import pymupdf
from PIL import Image
from openai import AsyncOpenAI
import pytesseract
import asyncio
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

app = FastAPI(title="PDF OCR Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class General_Info(BaseModel):
    invoice_number: str
    invoice_date: str
    importer: str
    seller: str
    origin_country: str
    origin_port: str
    origin_airport: str
    destination_country: str
    destination_port: str
    destination_airport: str
    invoice_currency: str
    invoice_incoterm: str
    total_goods_value: str
    total_fob_value: str
    freight_charges: str
    insurance_charges: str
    total_weight: str

class Import_Export(BaseModel):
    article_description: str
    article_composition: str
    hs_code: str
    quantity: str
    unit: str
    unit_price: str
    total_price: str

class First_Page(BaseModel):
    general_info: General_Info
    import_export: List[Import_Export]
    
class Second_Page(BaseModel):
    import_export: List[Import_Export]

class FreightGeneralInfo(BaseModel):
    freight_quotation_number: str
    freight_date: str
    freight_agent: str
    importer: str
    shipment_country: str
    shipment_port: Optional[str] = None
    shipment_airport: Optional[str] = None
    destination_country: str
    destination_port: Optional[str] = None
    destination_airport: Optional[str] = None
    transport_mode: str
    transport_incoterm: str
    freight_quotation_currency: str
    fob_charges: float
    freight_charges: float
    insurance_charges: float
    total_freight_weight: float
    
class CustomsDeclaration(BaseModel):
    article_number: str
    hs_code: str
    article_description: str
    origin: str
    quantity: float
    unit: str
    unit_fob_price: float
    total_fob_price: float    # multiply unit_fob_price by quantity

class CustomsGeneralInfo(BaseModel):
    import_declaration_number: str
    import_declaration_date: str
    customs_report_number: str
    customs_date: str
    invoice_date: str
    importer: str
    exporter: str
    origin_country: str
    destination_country: str
    invoice_currency: str
    invoice_incoterm: str
    total_goods_value: float
    total_fob_value: float
    unit_net_weight:  None
    total_net_weight: None
    unit_brut_weight: None
    total_brut_weight: None


class CustomsFreightGeneralInfo(BaseModel):
    transport_document_number: str
    shipment_country: str
    shipment_port: Optional[str] = None
    shipment_airport: Optional[str] = None
    destination_country: str
    destination_port: Optional[str] = None
    destination_airport: Optional[str] = None
    transport_mode: str
    containers: Optional[str] = None
    freight_charges: float
    insurance_charges: float
    other_charges: float
    total_freight_value: float
    total_brut_weight_kg: float
    total_net_weight_kg: float
    total_packages: int
    
class CustomsDeclaration_First_Page(BaseModel):
    custom_general_info: CustomsGeneralInfo
    freight_general_info: CustomsFreightGeneralInfo
    customs_declaration: List[CustomsDeclaration]
    
class CustomsDeclaration_Second_Page(BaseModel):
    customs_declaration: List[CustomsDeclaration] 
    

@app.get("/")
def read_root():
    return {"message": "PDF OCR Extractor API is running"}

def convert_pdf_to_images(file_content, dpi=300):
    """Convert PDF pages to PIL images."""
    doc = pymupdf.open(stream=file_content, filetype="pdf")
    images = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append((page_num + 1, img))
    doc.close()
    return images

def extract_text_with_ocr(img, page_num):
    """Extract text from a PIL image using Tesseract OCR."""
    logger.info(f"Starting OCR text extraction for page {page_num}")
    try:
        extracted_text = pytesseract.image_to_string(img)
        logger.info(f"Completed OCR text extraction for page {page_num}")
        return {"page": page_num, "text": extracted_text}
    except Exception as e:
        logger.error(f"Error extracting text from page {page_num}: {str(e)}")
        return {"page": page_num, "error": str(e)}

async def process_pdf_pages(file_content: bytes):
    """Process PDF pages and extract text."""
    start_time = time.time()
    images = convert_pdf_to_images(file_content)
    logger.info(f"Converted PDF to {len(images)} images")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_text_with_ocr, img, page_num): page_num 
                   for page_num, img in images}
        
        for future in concurrent.futures.as_completed(futures):
            page_num = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Exception processing page {page_num}: {str(e)}")
                results.append({"page": page_num, "error": str(e)})
    
    results.sort(key=lambda x: x["page"])
    logger.info(f"Completed OCR extraction for all pages in {time.time() - start_time:.2f} seconds")
    return results

async def analyze_with_gpt(extracted_data: List[Dict]):
    """Send the extracted text data to GPT for analysis."""
    start_time = time.time()
    logger.info("Starting GPT analysis of extracted text")
    logger.info("Length of extracted data: %s", len(extracted_data))

    def parse_response(response: str, page_number: int):
        parsed_data = json.loads(response)
        if page_number == 1:
            return First_Page(**parsed_data)
        else:
            return Second_Page(**parsed_data)
    
    async def analyze_page(item):
        page_text = item['text']
        page_number = item['page']
        try:
            if page_number == 1:
                response = await client.beta.chat.completions.parse(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": f"You are an expert Customs Officer. Convert raw data into structured JSON based on the provided schema."},
                        {"role": "user", "content": f" Convert raw data into structured JSON based on the provided schema. Page {page_number}: {page_text}\nConvert the data into structured JSON using the provided data. Include all data, even if duplicate. Do not make up any information."}
                    ],
                    temperature=0,
                    response_format=First_Page
                )
                
                logger.info(f"Type of response: {type(response.choices[0].message.content)}")
                logger.info(f"Response content: {response.choices[0].message.content}")
                final_response  = parse_response(response.choices[0].message.content, page_number)
                return {
                    "page": page_number,
                    "general_info": final_response.general_info,
                    "import_export": final_response.import_export
                }
            else:
                response = await client.beta.chat.completions.parse(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": f"You are an expert Customs Officer. Convert raw data into structured JSON based on the provided schema:"},
                        {"role": "user", "content": f"Page {page_number}: {page_text}\nConvert the data into structured JSON using the provided schema. Include all data, even if duplicate. Do not make up any information."}
                    ],
                    temperature=0,
                    response_format=Second_Page
                )
                final_response  = parse_response(response.choices[0].message.content,page_number)
                return {
                    "page": page_number,
                    "import_export": final_response.import_export
                }
        except Exception as e:
            logger.error(f"Error processing page {page_number}: {str(e)}")
            return {
                "page": page_number,
                "error": str(e)
            }

    try:
        if len(extracted_data) > 1:
            # Run all page analyses concurrently
            tasks = [analyze_page(item) for item in extracted_data]
            responses = await asyncio.gather(*tasks)
            logger.info(f"Completed GPT analysis for all pages in {time.time() - start_time:.2f} seconds")
            return responses
        else:
            combined_text = "\n".join([f"Page {item['page']}: {item['text']}" for item in extracted_data])
            response = await client.beta.chat.completions.parse(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {"role": "system", "content": "You are an expert Customs Officer. Convert raw data into structured JSON."},
                    {"role": "user", "content": combined_text + "\n Please don't skip any information. List all the data even if it's duplicate. Don't make up any information."}
                ],
                temperature=0,
                response_format=First_Page
            )
            logger.info(f"Type of response: {type(response.choices[0].message.content)}")
            logger.info(f"Response content: {response.choices[0].message.content}")
            logger.info(f"Completed GPT analysis in {time.time() - start_time:.2f} seconds")
            
            final_response  = parse_response(response.choices[0].message.content, 1)
            
            return {
                    "page": 1,
                    "general_info": final_response.general_info,
                    "import_export": final_response.import_export
                }
    except Exception as e:
        logger.error(f"Error in GPT analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GPT analysis failed: {str(e)}")



async def analyze_with_gpt_custom_declaration(extracted_data: List[Dict]):
    """Send the extracted text data to GPT for analysis."""
    start_time = time.time()
    logger.info("Starting GPT analysis of extracted text")
    logger.info("Length of extracted data: %s", len(extracted_data))

    def parse_response(response: str, page_number: int):
        parsed_data = json.loads(response)
        if page_number == 1:
            return CustomsDeclaration_First_Page(**parsed_data)
        else:
            return CustomsDeclaration_Second_Page(**parsed_data)
    
    async def analyze_page(item):
        page_text = item['text']
        page_number = item['page']
        try:
            if page_number == 1:
                response = await client.beta.chat.completions.parse(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": f"You are an expert Customs Officer. Convert raw data into structured JSON based on the provided schema."},
                        {"role": "user", "content": f""" Convert raw data into structured JSON based on the provided schema. Page {page_number}: {page_text}\nConvert the data into structured JSON using the provided data. Include all data, even if duplicate. Do not make up any information.
                         If there is no total_fob_value, its quantity multiplied by unit_fob_price should be used as total_fob_value."""}
                    ],
                    temperature=0,
                    response_format=CustomsDeclaration_First_Page
                )
                
                logger.info(f"Type of response: {type(response.choices[0].message.content)}")
                logger.info(f"Response content: {response.choices[0].message.content}")
                final_response  = parse_response(response.choices[0].message.content, page_number)
                return {
                    "page": page_number,
                    "custom_general_info": final_response.custom_general_info,
                    "freight_general_info": final_response.freight_general_info,
                    "customs_declaration": final_response.customs_declaration
                }
            else:
                response = await client.beta.chat.completions.parse(
                    model="gpt-4.1-2025-04-14",
                    messages=[
                        {"role": "system", "content": f"You are an expert Customs Officer. Convert raw data into structured JSON based on the provided schema:"},
                        {"role": "user", "content": f"Page {page_number}: {page_text}\nConvert the data into structured JSON using the provided schema. Include all data, even if duplicate. Do not make up any information.If there is no total_fob_value, its quantity multiplied by unit_fob_price should be used as total_fob_value."}
                    ],
                    temperature=0,
                    response_format=CustomsDeclaration_Second_Page
                )
                final_response  = parse_response(response.choices[0].message.content,page_number)
                return {
                    "page": page_number,
                    "customs_declaration": final_response.customs_declaration
                }
        except Exception as e:
            logger.error(f"Error processing page {page_number}: {str(e)}")
            return {
                "page": page_number,
                "error": str(e)
            }

    try:
        if len(extracted_data) > 1:
            # Run all page analyses concurrently
            
            filtered_data = []
            for item in extracted_data:
                filtered_data.append(item)
                if "Opinion valeur" in item['text'] :
                    logger.info(f"Found 'Opinion valeur et droits et taxes (XAF)' on page {item['page']}. Including this page and stopping further pages.")
                    break
            logger.info(f"Processing {len(filtered_data)} pages out of {len(extracted_data)}")

            # Create tasks only for the filtered data
            tasks = [analyze_page(item) for item in filtered_data]
            responses = await asyncio.gather(*tasks)
            logger.info(f"Completed GPT analysis for all pages in {time.time() - start_time:.2f} seconds")
            return responses
        
        else:
            combined_text = "\n".join([f"Page {item['page']}: {item['text']}" for item in extracted_data])
            response = await client.beta.chat.completions.parse(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {"role": "system", "content": "You are an expert Customs Officer. Convert raw data into structured JSON."},
                    {"role": "user", "content": combined_text + "\n Please don't skip any information. List all the data even if it's duplicate. Don't make up any information. If there is no total_fob_value, its quantity multiplied by unit_fob_price should be used as total_fob_value."}
                ],
                temperature=0,
                response_format=CustomsDeclaration_First_Page
            )
            logger.info(f"Type of response: {type(response.choices[0].message.content)}")
            logger.info(f"Response content: {response.choices[0].message.content}")
            logger.info(f"Completed GPT analysis in {time.time() - start_time:.2f} seconds")
            
            final_response  = parse_response(response.choices[0].message.content, 1)
            return {
                    "page": 1,
                    "custom_general_info": final_response.custom_general_info,
                    "freight_general_info": final_response.freight_general_info,
                    "customs_declaration": final_response.customs_declaration
                }
    except Exception as e:
        logger.error(f"Error in GPT analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GPT analysis failed: {str(e)}")
    

async def analyze_with_gpt_fret(extracted_data: List[Dict]):
    """Send the extracted text data to GPT for analysis."""
    start_time = time.time()
    logger.info("Starting GPT analysis of extracted text")
    logger.info("Length of extracted data: %s", len(extracted_data))

    def parse_response(response: str):
        parsed_data = json.loads(response)
        return FreightGeneralInfo(**parsed_data)

    try:
        combined_text = "\n".join([f"Page {item['page']}: {item['text']}" for item in extracted_data])
        response = await client.beta.chat.completions.parse(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are an expert Customs Officer. Convert raw data into structured JSON."},
                {"role": "user", "content": combined_text + "\n Please don't skip any information. List all the data even if it's duplicate. Don't make up any information."}
            ],
            temperature=0,
            response_format=FreightGeneralInfo
        )
        logger.info(f"Type of response: {type(response.choices[0].message.content)}")
        logger.info(f"Response content: {response.choices[0].message.content}")
        logger.info(f"Completed GPT analysis in {time.time() - start_time:.2f} seconds")
        
        final_response  = parse_response(response.choices[0].message.content)
        return {
                    "page": 1,
                    "FreightGeneralInfo": final_response,
                }
    except Exception as e:
        logger.error(f"Error in GPT analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GPT analysis failed: {str(e)}")
    

@app.post("/upload-commercial-invoice/", response_model=dict)
async def upload_and_process_pdf_pages(file: UploadFile = File(...)):
    logger.info(f"Received file upload for page analysis: {file.filename}")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes of PDF content")
        
        extraction_results = await process_pdf_pages(file_content)
        success_results = [r for r in extraction_results if "text" in r]
        error_results = [r for r in extraction_results if "error" in r]
        
        if not success_results:
            raise HTTPException(status_code=400, detail="No text extracted successfully.")
        
        final_summary = await analyze_with_gpt(success_results)
        logger.info("Final summary generated successfully.")
        
        return {
            "filename": file.filename,
            "pages": [{"page": r["page"], "text": r["text"]} for r in success_results],
            "errors": [{"page": r["page"], "error": r["error"]} for r in error_results],
            "final_summary": final_summary
        }
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    
    
@app.post("/upload-fret/", response_model=dict)
async def upload_and_process_pdf_pages(file: UploadFile = File(...)):
    logger.info(f"Received file upload for page analysis: {file.filename}")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes of PDF content")
        
        extraction_results = await process_pdf_pages(file_content)
        success_results = [r for r in extraction_results if "text" in r]
        error_results = [r for r in extraction_results if "error" in r]
        
        if not success_results:
            raise HTTPException(status_code=400, detail="No text extracted successfully.")
        
        final_summary = await analyze_with_gpt_fret(success_results)
        logger.info("Final summary generated successfully.")
        
        return {
            "filename": file.filename,
            "pages": [{"page": r["page"], "text": r["text"]} for r in success_results],
            "errors": [{"page": r["page"], "error": r["error"]} for r in error_results],
            "final_summary": final_summary
        }
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    

@app.post("/upload-custom-declaration/", response_model=dict)
async def upload_and_process_pdf_pages(file: UploadFile = File(...)):
    logger.info(f"Received file upload for page analysis: {file.filename}")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        file_content = await file.read()
        logger.info(f"Read {len(file_content)} bytes of PDF content")
        
        extraction_results = await process_pdf_pages(file_content)
        success_results = [r for r in extraction_results if "text" in r]
        error_results = [r for r in extraction_results if "error" in r]
        
        if not success_results:
            raise HTTPException(status_code=400, detail="No text extracted successfully.")
        
        final_summary = await analyze_with_gpt_custom_declaration(success_results)
        logger.info("Final summary generated successfully.")
        
        return {
            "filename": file.filename,
            "pages": [{"page": r["page"], "text": r["text"]} for r in success_results],
            "errors": [{"page": r["page"], "error": r["error"]} for r in error_results],
            "final_summary": final_summary
        }
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
    
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)