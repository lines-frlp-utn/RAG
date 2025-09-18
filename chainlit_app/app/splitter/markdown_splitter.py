from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def split_text_with_langchain(texts, chunk_size=1000, chunk_overlap=200):
    """
    Divide el texto en fragmentos utilizando RecursiveCharacterTextSplitter de LangChain.

    Args:
        texts (list): Lista de textos a dividir.
        chunk_size (int): Tamaño máximo de cada fragmento.
        chunk_overlap (int): Cantidad de superposición entre fragmentos.

    Returns:
        list: Lista de fragmentos de texto.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[".", "!", "?", "\n\n", "\n", "\t", ",", ";", ":"],
    )

    # Usar comprensión de listas para dividir texto
    return [chunk for text in texts for chunk in text_splitter.split_text(text)]


def split_markdown_text(text, max_length=1000, chunk_overlap=200) -> list[str]:
    """
        Divide texto Markdown en fragmentos manteniendo la estructura de encabezados.

        Primero divide el texto por encabezados Markdown (#, ##, ###, etc.) y luego
        fragmenta los bloques que exceden el tamaño máximo usando división recursiva.

        Args:
            text (str): Texto Markdown a dividir en fragmentos.
            max_length (int, optional): Tamaño máximo permitido para cada fragmento. 
                                        Por defecto 1000 caracteres.
            chunk_overlap (int, optional):  Cantidad de caracteres de superposición 
                                            entre fragmentos adyacentes. Por defecto 200.

        Returns:
            list[str]:  Lista de fragmentos de texto procesados, respetando la estructura
                        de encabezados y los límites de tamaño.
    """
    
    # Definir los encabezados de Markdown para dividir
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    # Dividir el texto en fragmentos basados en encabezados
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)

    # Primero, dividir el texto en fragmentos basados en encabezados
    chunks = markdown_splitter.split_text(text)
    markdown_chunks = [chunk.page_content for chunk in chunks]

    # Imprimir información de depuración
    for i, chunk in enumerate(markdown_chunks):
        print(f"Chunk {i + 1} length: {len(chunk)}")

    # Aplicar RecursiveCharacterTextSplitter para fragmentos más pequeños si es necesario
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    # Crear la lista final de fragmentos
    final_chunks = []
    for chunk in markdown_chunks:
        if len(chunk) > max_length:
            smaller_chunks = text_splitter.split_text(chunk)
            final_chunks.extend(smaller_chunks)
        else:
            final_chunks.append(chunk)

    # Imprimir información de depuración final
    print(f"Final chunks: {len(final_chunks)}")
    for i, chunk in enumerate(final_chunks):
        print(f"Chunk {i + 1} length: {len(chunk)}")

    return final_chunks
