# Proyecto final - IBM2020
## Caracterización del flujo sanguíneo cerebral

### Manejo de datos
En primer lugar, desde el repositorio de [BrainMeshLibrary](https://github.com/NeuroJSON/BrainMeshLibrary) se obtuvieron modelos de malla de cabezas, segmentados por regiones cerebrales (Scalp, Skull, CSF, GM, WM), en particular, se utilizó el modelo 05. Estos modelos se encuentran en formato `JSON/JMesh` en la carpeta `/data`. En primer lugar, fueron convertidos a formato `msh` utilizando el código `main.m`, los archivos obtenidos se encuentran el carpeta `/msh_files`. Luego, aquellos archivos fueron abiertos en el _software_ **GMesh** para convertirlos al formato `inp`, los que fueron almacenados en la carpeta `/inp_files`. Finalmente, estos últimos fueron abiertos en el _software_ de simulación **FEBio**, los modelos se encuentran en la carpeta `/model`.

#### Hecho con :heart: por Iván Vergara Lam