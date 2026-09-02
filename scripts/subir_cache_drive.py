"""
Sube los artefactos precomputados (carpeta cache/, generada por
precompute_cache.py) a Google Drive usando el mismo service account de
credentials.json, y los deja públicos ("cualquiera con el link puede ver")
para que la app los descargue con gdown igual que los CSV crudos.

Imprime al final el diccionario de IDs para pegar en app_tam_v2.py.

Uso:
    python scripts/subir_cache_drive.py
"""
import os
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import AuthorizedSession

SCOPES = ["https://www.googleapis.com/auth/drive"]
CACHE_DIR = "cache"
ARCHIVOS = [
    "obras.parquet",
    "inter.parquet",
    "pop_fac.parquet",
    "pop_prog.parquet",
    "pop_gen.parquet",
    "tfidf_matrix.npz",
    "tfidf_vectorizer.joblib",
]


def main():
    creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
    session = AuthorizedSession(creds)

    ids = {}
    for nombre in ARCHIVOS:
        ruta = os.path.join(CACHE_DIR, nombre)
        if not os.path.exists(ruta):
            print(f"AVISO: no existe {ruta}, se omite")
            continue
        size_mb = os.path.getsize(ruta) / 1024 / 1024
        print(f"Subiendo {nombre} ({size_mb:.1f} MB)...")

        metadata = {"name": nombre}
        with open(ruta, "rb") as f:
            files = {
                "metadata": (None, __import__("json").dumps(metadata), "application/json"),
                "file": (nombre, f, "application/octet-stream"),
            }
            r = session.post(
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id",
                files=files,
            )
        r.raise_for_status()
        file_id = r.json()["id"]

        # Hacerlo público (cualquiera con el link, solo lectura) para que gdown lo descargue sin auth.
        r2 = session.post(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
            json={"role": "reader", "type": "anyone"},
        )
        r2.raise_for_status()

        ids[nombre] = file_id
        print(f"  -> id: {file_id}")

    print("\n--- IDs para pegar en app_tam_v2.py ---")
    for nombre, file_id in ids.items():
        const_name = "ID_" + nombre.rsplit(".", 1)[0].upper()
        print(f'{const_name} = "{file_id}"  # {nombre}')


if __name__ == "__main__":
    main()
