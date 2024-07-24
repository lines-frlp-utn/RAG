from pymilvus import MilvusClient
import fastapi


client = MilvusClient("./database/milvus_demo.db")

def upload_pdf_to_vector_db(dataWithEmbeddings: dict | list[dict], collection_name):

    if client.has_collection(collection_name=collection_name) == False:
        client.create_collection(
            collection_name = collection_name,
            metric_type = "COSINE", #revisar
            schema = None, #revisar
            )
        print("Collection created")

    result = client.insert(
        collection_name = collection_name,
        data = dataWithEmbeddings,
    )

    print("Docs uploaded to Milvus")
    print(result)


def get_context_with_filters(collection_name, theme, subtheme, query: list):
    respuesta = client.search(
        data=[query], ##Valor que buscamos
        anns_field="embedding", ##Campo con el que comparamos el embedding de consulta
        param={"metric_type": "L2",
            "params": {"nprobe":2}}, ##Definimos el tipo de metrica | nprobe 2 => 2 centroids INVESTIGAR ESTOS PARAMETROS POR AHORA LO DEJO DEFAULT
        limit=5, ##Limita a 3 resultados por busqueda
        output_fields=["texto"] ##Campo que queremos que devuelva
    )
