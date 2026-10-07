"""
Precomputa los artefactos de la TERCERA iteracion (directorio institucional
completo + historico extendido de prestamos de tercera_iteracion/), siguiendo
el mismo patron que scripts/precompute_cache.py.

Proteccion de datos (acordada explicitamente con la autora, ver conversacion):
  - Las columnas de identificacion personal real (nombre, apellido, correo,
    telefono, direccion, fecha de nacimiento, nacionalidad, cedula/barcode)
    NUNCA se cargan a memoria: el pd.read_csv(usecols=...) de mas abajo ni
    siquiera las incluye. No es una promesa de "no las voy a imprimir", es
    que ese dato no entra al proceso.
  - assert_sin_pii() es una red de seguridad adicional: si alguna de esas
    columnas apareciera en el DataFrame por cualquier motivo, el script se
    detiene de inmediato en vez de seguir procesando.
  - El "ID de estudiante" (customFields.idSistemaExterno) que el participante
    escribe en la app es un identificador personal real. Se usa UNICAMENTE
    para encontrar el `id` interno (UUID ya existente, no mas sensible que
    el resto del pipeline) y se descarta de inmediato -- no se guarda en
    ningun artefacto junto con facultad/programa/historial. El unico archivo
    que lo contiene es cache/estudiante_lookup.parquet, con exactamente dos
    columnas (id_estudiante, id), nada mas.

Reusa el catalogo + TF-IDF ya calculado en la iteracion 2
(cache/obras.parquet, cache/tfidf_matrix.npz, cache/tfidf_vectorizer.joblib):
el cruce exploratorio ya confirmo que 78.9% de los item.instanceId de este
nuevo historico tienen su obra en ese catalogo, asi que no hace falta
reprocesar Biblioteca_General.csv de nuevo.

Corre localmente (desde la raiz del proyecto):
    python scripts/precompute_cache_v3.py

Para probar contra datos sinteticos antes de correr sobre los reales, se
puede apuntar a otra carpeta con la variable de entorno:
    CARPETA_TERCERA_ITERACION=ruta\\de\\prueba python scripts/precompute_cache_v3.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from precompute_cache import MAPEO_FAC, nt  # reusa el mismo mapeo/normalizador que la iteracion 2

CARPETA = os.environ.get("CARPETA_TERCERA_ITERACION", "tercera_iteracion")
OUT_DIR = os.environ.get("CACHE_OUT_DIR", "cache")

PATH_USUARIOS = os.path.join(CARPETA, "Data_Total_Usuarios_Folio_Javeriana.csv")
PATH_PRESTAMOS = os.path.join(CARPETA, "Total_Historico_Completo_Prestamos_Folio.csv")

# Columnas permitidas -- el resto nunca se carga a memoria.
USUARIOS_COLS = [
    "id", "customFields.idSistemaExterno", "patronGroup",
    "customFields.facultad", "customFields.programa_2",
]
PRESTAMOS_COLS = [
    "userId", "item.instanceId", "loanDate", "action", "renewalCount",
    "patronGroupAtCheckout.name",
]

# Red de seguridad adicional (ver docstring). Nombres tal como aparecen en el
# esquema real de Data_Total_Usuarios_Folio_Javeriana.xlsx.
COLUMNAS_PROHIBIDAS = {
    "username", "externalSystemId", "barcode", "active", "type",
    "preferredEmailCommunication", "personal.lastName", "personal.firstName",
    "personal.email", "personal.phone", "personal.mobilePhone",
    "personal.addresses", "personal.preferredContactTypeId",
    "metadata.createdByUserId", "metadata.updatedByUserId",
    "customFields.correoPersonalizado", "enrollmentDate", "expirationDate",
    "personal.middleName", "customFields.sede", "customFields.curso",
    "customFields.notaSymphony", "customFields.nacionalidad",
    "meta.creation_date", "meta.last_login_date", "customFields.notas",
    "tags.tagList", "personal.dateOfBirth", "personal.preferredFirstName",
    "personal.pronouns", "createdDate", "updatedDate", "departments",
    "proxyFor", "customFields.genero", "customFields.codprograma",
}


def assert_sin_pii(df: pd.DataFrame, nombre: str):
    coladas = COLUMNAS_PROHIBIDAS & set(df.columns)
    assert not coladas, f"[SEGURIDAD] {nombre} tiene columnas prohibidas: {coladas}"


# Decodificacion de patronGroup (UUID -> tipo de usuario). Fuente: catalogo
# tercera_iteracion/Data_usuarios_grupos_Folio_Javeriana.xlsx (10 filas) + el
# cruce agregado patronGroup x patronGroupAtCheckout.name hecho en
# Exploracion_Tercera_Iteracion.ipynb (seccion 4.1). Los UUID sin resolver
# con certeza quedan fuera del dict (no se adivina) y caen al "" por defecto.
DECODE_PATRON_GROUP = {
    "aa345887-1902-4d7c-907b-a7cc9229e868": "Egresado",        # catalogo
    "0fe2c94d-32f6-44ee-811c-968ee51e0941": "Inactivo",        # catalogo
    "bc05bbdd-3d91-4cac-907b-36333f8ee675": "Estudiante",      # catalogo + confirmado por dato real
    "e7970ded-bac5-4f58-94b2-718ee9ffa159": "Estudiante",      # inferido (cruce, sede Cali)
    "876cca63-ca6f-4339-90ad-775ee55dabc8": "Academico",       # inferido (cruce, sede Bogota)
    "dd6342db-7050-47e1-bd0f-6440d69b2a92": "Administrativo",  # inferido (cruce, sede Bogota)
    "dc4c0e23-d27c-4dbd-959b-e3c242e71112": "Academico",       # inferido (cruce, sede Cali)
    "2c230d29-fa0c-4e8a-9193-c8abccbf1add": "Administrativo",  # catalogo (sede Cali)
    "7a93a94f-eaab-4448-a8d5-86eb54eb481b": "Convenio",        # catalogo
    "0a0d3ef6-d45b-44f0-bfd4-d0172bc507ba": "Especial",        # catalogo
    "6376321b-ddaa-48f8-a0f4-6e5f1a62aa7a": "Staff",           # catalogo
    # 91880a70-..., d93664c7-..., 65dc20d0-..., c74fc9e3-..., 373d5fb3-...,
    # 1133089e-..., ed0f4f28-...: sin resolver con certeza -> "".
}

# Equivalente de ACCIONES_VALIDAS de precompute_cache.py, mapeado a los
# nombres de "action" en este nuevo historico (14 categorias vs. 5).
ACCIONES_VALIDAS_V3 = {
    "checkedin", "checkedout", "renewed", "checkedOutThroughOverride",
    "renewedThroughOverride", "checkedInReturnedByPatron", "closedLoan",
    "checkedInFoundByLibrary",
}


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Leyendo usuarios desde {PATH_USUARIOS} (solo columnas permitidas)...")
    usuarios = pd.read_csv(PATH_USUARIOS, usecols=USUARIOS_COLS, low_memory=False)
    assert_sin_pii(usuarios, "usuarios")

    print(f"Leyendo prestamos desde {PATH_PRESTAMOS} (solo columnas permitidas)...")
    prestamos = pd.read_csv(PATH_PRESTAMOS, usecols=PRESTAMOS_COLS, low_memory=False)
    assert_sin_pii(prestamos, "prestamos")

    n_total = len(prestamos)

    # --- Filtro de sede: solo Bogota (decidido explicitamente) ---
    es_cali = prestamos["patronGroupAtCheckout.name"].str.contains("Cali", case=False, na=False)
    es_sede_desconocida = prestamos["patronGroupAtCheckout.name"].isna()
    prestamos = prestamos[~es_cali & ~es_sede_desconocida].copy()
    n_bogota = len(prestamos)

    # --- Filtro de acciones validas (equivalente a ACCIONES_VALIDAS) ---
    prestamos = prestamos[prestamos["action"].isin(ACCIONES_VALIDAS_V3)].copy()
    n_accion_valida = len(prestamos)

    # --- Fechas / recencia ---
    prestamos["loanDate"] = pd.to_datetime(prestamos["loanDate"], errors="coerce", utc=True)
    prestamos = prestamos.dropna(subset=["loanDate", "userId", "item.instanceId"]).copy()
    mf = prestamos["loanDate"].max()
    dias = (mf - prestamos["loanDate"]).dt.days
    peso_recencia = np.exp(-dias / 180)

    # --- Renovaciones como senal graduada (en vez del 0.5x binario de la
    # iteracion anterior, que solo distinguia "tuvo renovacion" si/no) ---
    renovaciones = prestamos["renewalCount"].fillna(0)
    peso_renovacion = 1 + 0.1 * renovaciones
    prestamos["peso_evento"] = peso_renovacion * (1 + peso_recencia)

    print("Agregando interacciones usuario-obra (id interno, nunca el ID que escribe el estudiante)...")
    inter_v3 = (
        prestamos.rename(columns={"userId": "id", "item.instanceId": "instance_id"})
        .groupby(["id", "instance_id"])
        .agg(peso=("peso_evento", "sum"), n=("action", "count"))
        .reset_index()
    )
    inter_v3["peso"] = np.log1p(inter_v3["peso"])

    # --- Perfil institucional limpio (facultad/programa normalizados + tipo
    # de usuario decodificado). customFields.genero NO se incluye: son
    # codigos opacos (opt_0/opt_1/opt_2) sin tabla de referencia confirmada
    # -- a diferencia del formulario actual (donde "Femenino"/"Masculino" es
    # texto real que puede coincidir lexicamente con el catalogo), un codigo
    # opt_N no aporta nada a un TF-IDF de perfil. Si se confirma el mapeo
    # opt_N -> genero con FOLIO/la universidad, se puede agregar despues.
    # fillna("") ANTES de nt()/map(): nt() hace str(valor), y str(NaN) es el
    # texto literal "nan" -- sin este fillna previo, los nulos de facultad o
    # programa quedarian como la palabra "nan" en vez de vacios.
    facultad_rellena = usuarios["customFields.facultad"].fillna("")
    usuarios["facultad_clean"] = facultad_rellena.map(MAPEO_FAC).fillna(facultad_rellena.apply(nt))
    usuarios["programa_clean"] = usuarios["customFields.programa_2"].fillna("").apply(nt)
    usuarios["tipo_usuario"] = usuarios["patronGroup"].map(DECODE_PATRON_GROUP).fillna("")

    perfil_v3 = usuarios[["id", "facultad_clean", "programa_clean", "tipo_usuario"]].copy()

    # --- Popularidad por programa y por facultad, ya escopada a Bogota
    # porque "prestamos" ya viene filtrado arriba ---
    inter_con_perfil = inter_v3.merge(
        usuarios[["id", "facultad_clean", "programa_clean"]], on="id", how="left"
    )
    pop_fac_v3 = (
        inter_con_perfil.groupby(["facultad_clean", "instance_id"])
        .agg(n_usuarios=("id", "nunique")).reset_index()
    )
    pop_fac_v3["score_pop"] = np.log1p(pop_fac_v3["n_usuarios"])

    pop_prog_v3 = (
        inter_con_perfil.groupby(["programa_clean", "instance_id"])
        .agg(n_usuarios=("id", "nunique")).reset_index()
    )
    pop_prog_v3["score_pop"] = np.log1p(pop_prog_v3["n_usuarios"])

    # --- Lookup minimo: SOLO lo necesario para traducir el "ID de estudiante"
    # que escribe el participante al id interno. No contiene facultad,
    # programa, tipo de usuario, ni ningun otro dato -- eso vive aparte en
    # perfil_v3.parquet, indexado por el id interno, nunca por el ID visible.
    lookup = (
        usuarios[["customFields.idSistemaExterno", "id"]]
        .dropna()
        .rename(columns={"customFields.idSistemaExterno": "id_estudiante"})
    )

    print(f"Guardando artefactos en {OUT_DIR}/ ...")
    inter_v3.to_parquet(f"{OUT_DIR}/inter_v3.parquet", index=False)
    perfil_v3.to_parquet(f"{OUT_DIR}/perfil_v3.parquet", index=False)
    pop_fac_v3.to_parquet(f"{OUT_DIR}/pop_fac_v3.parquet", index=False)
    pop_prog_v3.to_parquet(f"{OUT_DIR}/pop_prog_v3.parquet", index=False)
    lookup.to_parquet(f"{OUT_DIR}/estudiante_lookup.parquet", index=False)

    for fn in ["inter_v3.parquet", "perfil_v3.parquet", "pop_fac_v3.parquet",
               "pop_prog_v3.parquet", "estudiante_lookup.parquet"]:
        size_mb = os.path.getsize(f"{OUT_DIR}/{fn}") / 1024 / 1024
        print(f"  {fn}: {size_mb:.1f} MB")

    print(f"\nPrestamos totales: {n_total:,}")
    print(f"  Bogota confirmado: {n_bogota:,} ({n_bogota/n_total*100:.1f}%)")
    print(f"  ... con accion valida: {n_accion_valida:,} ({n_accion_valida/n_total*100:.1f}%)")
    print(f"Filas finales en inter_v3: {len(inter_v3):,}")
    print(f"Usuarios con lookup por ID de estudiante: {len(lookup):,} de {len(usuarios):,}")
    print(f"Listo en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
