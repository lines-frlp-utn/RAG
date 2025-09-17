```md
RAG/
├── .devcontainer/               # Configuración para entornos de desarrollo con Docker/VSCode
│   ├── compose.extend.yml       # Extensión de configuración docker-compose para devcontainer
│   └── devcontainer.json        # Configuración principal del devcontainer
├── chainlit_app/                # Aplicación principal Chainlit (RAG)
│   ├── .chainlit/               # Configuración y traducciones de Chainlit
│   │   ├── config.toml          # Configuración de Chainlit
│   │   └── translations/        # Traducciones de la interfaz Chainlit
│   │       ├── bn.json          # Bengalí
│   │       ├── en-US.json       # Inglés (EE.UU.)
│   │       ├── gu.json          # Gujarati
│   │       ├── he-IL.json       # Hebreo (Israel)
│   │       ├── hi.json          # Hindi
│   │       ├── ja.json          # Japonés
│   │       ├── kn.json          # Kannada
│   │       ├── ml.json          # Malayalam
│   │       ├── mr.json          # Marathi
│   │       ├── nl-NL.json       # Neerlandés (Países Bajos)
│   │       ├── nl.json          # Neerlandés
│   │       ├── ta.json          # Tamil
│   │       ├── te.json          # Telugu
│   │       └── zh-CN.json       # Chino simplificado
│   ├── .dockerignore            # Archivos ignorados por Docker
│   ├── app/                     # Código fuente principal de la app
│   │   ├── aim_tracker.py       # Módulo para seguimiento de objetivos
│   │   ├── auth.py              # Autenticación de usuarios
│   │   ├── config.py            # Configuración de la app
│   │   ├── databases.py         # Conexión y lógica de bases de datos
│   │   ├── embedding_generator.py # Generación de embeddings para textos
│   │   ├── main.py              # Punto de entrada principal de la app
│   │   ├── models.py            # Modelos de datos (Pydantic, etc.)
│   │   ├── parser.py            # Parseo de documentos/textos
│   │   ├── splitter/            # Lógica para dividir textos/documentos
│   │   │   ├── markdown_splitter.py # Splitter para Markdown
│   │   │   └── semantic_splitter.py # Splitter semántico
│   │   └── umap_results/        # Resultados de reducción dimensionalidad (UMAP)
│   │       ├── .gitkeep         # Archivo para mantener carpeta en Git
│   │       └── 1.png            # Imagen de resultados UMAP
│   ├── chainlit.md              # Documentación sobre Chainlit
│   ├── Dockerfile               # Dockerfile para construir la app
│   ├── pyproject.toml           # Configuración de dependencias Python
│   ├── requirements.txt         # Lista de dependencias Python
│   ├── tests/                   # Pruebas unitarias y de integración
│   │   ├── output_from_llamaparse.txt # Salida de pruebas de parseo
│   │   ├── pdfs_prueba/         # PDFs de ejemplo para pruebas
│   │   │   ├── algoritmos.pdf   # PDF de prueba: algoritmos
│   │   │   ├── bitcoin_es.pdf   # PDF de prueba: bitcoin en español
│   │   │   └── Ethereum.pdf     # PDF de prueba: Ethereum
│   │   ├── test_api_llamaParse.py # Test para API de parseo con LlamaParse
│   │   ├── test_splitter.py     # Test para splitter de textos
│   │   ├── test_upload_chroma_db.py # Test para subida a base Chroma
│   │   ├── test_upload_db.py    # Test para subida genérica a base de datos
│   │   ├── test_upload_milvus_db.py # Test para subida a Milvus
│   │   └── test_upload_milvus_db_with_embeddings.py # Test para subida a Milvus con embeddings
│   ├── translation              # Traducciones adicionales (estructura)
│   ├── translations/            # Traducciones (estructura)
│   │   └── en-US                # Traducción inglés EE.UU. (estructura)
│   └── uv.lock                  # Archivo de bloqueo de dependencias (uv)
├── docs/                        # Documentación y recursos gráficos
│   ├── 1.png                    # Imagen de documentación
│   ├── 2.png                    # Imagen de documentación
│   ├── 3.png                    # Imagen de documentación
│   ├── 4.png                    # Imagen de documentación
│   ├── 5.png                    # Imagen de documentación
│   └── extension.png            # Imagen de extensión
├── users/                       # Módulo de usuarios (API, Docker, dependencias)
│   ├── Dockerfile               # Dockerfile para usuarios
│   ├── main.py                  # Código fuente de usuarios
│   └── requirements.txt         # Dependencias para usuarios
├── vectordbs/                   # Implementaciones de bases de datos vectoriales
│   ├── Chroma/                  # Código y configuración para ChromaDB
│   │   ├── Dockerfile           # Dockerfile para ChromaDB
│   │   ├── main.py              # Código fuente para ChromaDB
│   │   └── requirements.txt     # Dependencias para ChromaDB
│   └── Milvus/                  # Código y configuración para MilvusDB
│       ├── Dockerfile           # Dockerfile para MilvusDB
│       ├── main.py              # Código fuente para MilvusDB
│       └── requirements.txt     # Dependencias para
├── .gitattributes               # Configuración de atributos para Git
├── .gitignore                   # Archivos y carpetas ignorados por Git
├── compose.base.yml             # Configuración base de docker-compose
├── compose.chroma.yml           # Configuración docker-compose para ChromaDB
├── compose.milvus.yml           # Configuración docker-compose para MilvusDB
├── dev.Dockerfile               # Dockerfile para entorno de desarrollo
├── pyproject.toml               # Configuración de dependencias Python (raíz)
├── README.md                    # Documentación principal del proyecto
└── Tree.md                      # Estructura de diretorios
```