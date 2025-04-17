# Proyecto RAG
### Cómo ejecutar en un dev container

1. Instalar la extensión de Dev Containers.

    <img src="docs/extension.png" alt="Extensión Dev Containers" style="max-width: 500px; max-height: 500px;">

2. Para entrar al Dev Container, utiliza el atajo `Ctrl + Shift + P` y busca "Dev Containers: Rebuild and Reopen in Container".

    <img src="docs/2.png" alt="Rebuild and Reopen" style="max-width: 500px; max-height: 500px;">

3. Dentro del contenedor, utiliza el depurador de Python para ejecutar la aplicación de Chainlit.

    <img src="docs/3.png" alt="Depurador de Python" style="max-width: 500px; max-height: 500px;">

4. Si realizas cambios en el archivo `docker-compose`, al entrar nuevamente al Dev Container, haz clic en el botón azul y selecciona "Reconstruir contenedor". Para salir del Dev Container, reabre la carpeta local.

    <img src="docs/4.png" alt="Reconstruir contenedor" style="max-width: 500px; max-height: 500px;">

5. Para seleccionar qué contenedores se inician al usar el Dev Container, edita el archivo `devcontainer.json` y descomenta los contenedores necesarios.

    <img src="docs/5.png" alt="Editar devcontainer.json" style="max-width: 500px; max-height: 500px;">

6. Las variables de entorno del Dev Container se pueden modificar en `.devcontainer/compose.extend.yml`.
