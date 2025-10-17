import tempfile
from pathlib import Path

import chainlit as cl
from app.config import conf as cfg
from app.databases import post_embeddings
from app.embedding_generator import embedding_generator
from app.parser import extract_text_from_pdf
from app.splitter.markdown_splitter import split_markdown_text as markdown_split
from chainlit.utils import mount_chainlit
from fastapi import FastAPI, UploadFile
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=".chainlit")
main_app = FastAPI()


@main_app.get("/upload-pdf")
def read_main(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@main_app.post("/upload-pdf")
async def upload_pdf(file: UploadFile):
    if file:
        try:
            # Save the uploaded file to a temporary location
            print(f"Guardando archivo temporal `{file.filename}`...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name

            # Extraer el texto del PDF
            print(f"Extrayendo texto de `{file.filename}`...")
            text = extract_text_from_pdf(tmp_path)

            # Splittear el texto en chunks semánticos
            print(f"Splitteando texto de `{file.filename}`...")
            chunks = markdown_split(text)

            # Generar los embeddings de los chunks
            print(f"Generando embeddings de `{file.filename}`...")
            embeddings = await cl.make_async(embedding_generator.get_embeddings)(chunks)

            # Formatear y cargar los embeddings en la base de datos
            print(f"Formateando embeddings de `{file.filename}`...")
            embeddings_data = await cl.make_async(embedding_generator.format_for_database)(
                embeddings, chunks
            )
            print("Embeddings formateados")
            match cfg.PROJECT_ENV:
                case "default":
                    collection_name = "prueba_lines"
                case "chat-lines":
                    collection_name = "chat_lines"
                case "chat-frlp":
                    collection_name = "chat_frlp"
            result = await cl.make_async(post_embeddings)(
                collection_name=collection_name, dataWithEmbeddings=embeddings_data
            )
            print(f"Archivo `{file.filename}` cargado exitosamente, `{result}`")

            message = f"PDF `{file.filename}` uploaded and processed successfully."

            # Clean up the temporary file
            Path(tmp_path).unlink()

        except Exception as e:
            message = f"Error procesando el archivo `{file.filename}`: {str(e)}"
            print(message)
    return {"message": message}


mount_chainlit(app=main_app, target="app/cl_app.py", path="/")
