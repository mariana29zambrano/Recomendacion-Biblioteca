"""
Precomputa el catálogo limpio + la matriz TF-IDF del Modelo 1 (rec_A/rec_C en
app_tam_v2.py) y los guarda como archivos listos para cargar (Parquet + npz +
joblib), en vez de que la app reprocese los CSV crudos (470MB + 64MB) en cada
arranque — eso es lo que hacía lento el arranque en frío y presionaba la RAM
del plan gratuito de Streamlit Cloud (1 GB) al punto de caerse con varios
usuarios simultáneos.

Correr localmente cuando cambien los CSV fuente:
    python scripts/precompute_cache.py

Duplica a propósito las constantes de columnas/mapeos de app_tam_v2.py (no se
puede importar ese módulo directo: llama a st.set_page_config() al cargarse,
lo cual falla fuera de `streamlit run`). Si cambian esas constantes o la
lógica de limpieza en app_tam_v2.py, hay que replicar el cambio acá también
y volver a correr este script.

Después de correrlo, subir los artefactos resultantes (carpeta cache/) a
Google Drive con scripts/subir_cache_drive.py.
"""
import json, re, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from unidecode import unidecode

COL_INSTANCE_ID = "Instances - Instance UUID"
COL_TITLE       = "Instances - Index title"
COL_CONTRIBUTORS= "Instances - Contributors"
COL_SUBJECTS    = "Instances - Subject headings"
COL_PUBLICATION = "Instances - Publication"
COL_CALL_NUMBER = "Items - Item call number"
COL_MTYPE       = "Material type - Name"
COL_LOCATION    = "Effective location - Name"
COL_BARCODE     = "Items - Barcode"

LOAN_USER    = "ID Usuario"
LOAN_BARCODE = "Item barcode"
LOAN_DATE    = "Date"
LOAN_ACTION  = "Circ action"
LOAN_PROFILE = "Perfil"
LOAN_PROGRAM = "Programa"
LOAN_FACULTY = "Facultad"
PERFIL_ESTUDIANTE = "Bogota Estudiantes"
ACCIONES_VALIDAS = ["Checked out","Checked out through override","Renewed","Renewed through override"]

MAPEO_FAC = {
    "FACULTAD DE ARQUITECTURA Y DISENO":"Facultad de Arquitectura y Diseño","DECANATURA DE FACULTAD DE ARQUITECTURA Y DISENO":"Facultad de Arquitectura y Diseño",
    "FACULTAD DE ARTES":"Facultad de Artes","DECANATURA DE FACULTAD DE ARTES":"Facultad de Artes",
    "FACULTAD DE CIENCIAS":"Facultad de Ciencias","DECANATURA DE FACULTAD DE CIENCIAS":"Facultad de Ciencias",
    "FACULTAD DE CIENCIAS ECONOMICAS Y ADMINISTRATIVAS":"Facultad de Ciencias Económicas y Administrativas","DECANATURA DE FACULTAD DE CIENCIAS ECONOMICAS Y ADMINISTRATIVAS":"Facultad de Ciencias Económicas y Administrativas",
    "FACULTAD DE CIENCIAS JURIDICAS":"Facultad de Ciencias Jurídicas","DECANATURA DE FACULTAD DE CIENCIAS JURIDICAS":"Facultad de Ciencias Jurídicas",
    "FACULTAD DE CIENCIAS POLITICAS Y RELACIONES INTERNACIONALES":"Facultad de Ciencias Políticas y Relaciones Internacionales","DECANATURA DE FACULTAD DE CIENCIAS POLITICAS Y RELACIONES INTERNACIONALES":"Facultad de Ciencias Políticas y Relaciones Internacionales",
    "FACULTAD DE CIENCIAS SOCIALES":"Facultad de Ciencias Sociales","DECANATURA DE FACULTAD DE CIENCIAS SOCIALES":"Facultad de Ciencias Sociales",
    "FACULTAD DE COMUNICACION Y LENGUAJE":"Facultad de Comunicación y Lenguaje","DECANATURA DE FACULTAD DE COMUNICACION Y LENGUAJE":"Facultad de Comunicación y Lenguaje",
    "FACULTAD DE DERECHO CANONICO":"Facultad de Derecho Canónico","DECANATURA DE FACULTAD DE DERECHO CANONICO":"Facultad de Derecho Canónico",
    "FACULTAD DE EDUCACION":"Facultad de Educación","DECANATURA DE FACULTAD DE EDUCACION":"Facultad de Educación",
    "FACULTAD DE ENFERMERIA":"Facultad de Enfermería","DECANATURA DE FACULTAD DE ENFERMERIA":"Facultad de Enfermería",
    "FACULTAD DE ESTUDIOS AMBIENTALES Y RURALES":"Facultad de Estudios Ambientales y Rurales","DECANATURA DE FACULTAD DE ESTUDIOS AMBIENTALES Y RURALES":"Facultad de Estudios Ambientales y Rurales",
    "FACULTAD DE FILOSOFIA":"Facultad de Filosofía","DECANATURA DE FACULTAD DE FILOSOFIA":"Facultad de Filosofía",
    "FACULTAD DE INGENIERIA":"Facultad de Ingeniería","DECANATURA DE FACULTAD DE INGENIERIA":"Facultad de Ingeniería",
    "FACULTAD DE MEDICINA":"Facultad de Medicina","DECANATURA DE FACULTAD DE MEDICINA":"Facultad de Medicina",
    "FACULTAD DE ODONTOLOGIA":"Facultad de Odontología","DECANATURA DE FACULTAD DE ODONTOLOGIA":"Facultad de Odontología",
    "FACULTAD DE PSICOLOGIA":"Facultad de Psicología","DECANATURA DE FACULTAD DE PSICOLOGIA":"Facultad de Psicología",
    "FACULTAD DE TEOLOGIA":"Facultad de Teología","DECANATURA DE FACULTAD DE TEOLOGIA":"Facultad de Teología",
}

def nt(t):
    t=str(t).lower(); t=unidecode(t); t=re.sub(r"[^a-z0-9\s]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

def contrib(v):
    try:
        p=json.loads(str(v))
        if isinstance(p,list): return " ".join([i.get("name","") for i in p if isinstance(i,dict)])
    except: pass
    return str(v) if v and str(v) not in ["","nan"] else ""

def publication_text(v):
    if not v or str(v).strip() in ["","nan"]: return ""
    try:
        p=json.loads(str(v))
        if isinstance(p,list):
            parts=[]
            for item in p:
                if isinstance(item,dict): parts.extend([str(x) for x in item.values() if x])
            return " ".join(parts)
    except: pass
    return str(v)

def ju(s):
    vals=s.dropna().astype(str).unique()
    return " ; ".join([v for v in vals if v.strip() not in ["","nan"]][:20])


def main():
    t0 = time.time()
    out_dir = "cache"
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("Leyendo Biblioteca_General.csv...")
    col = pd.read_csv("data/Biblioteca_General.csv", low_memory=False)
    print("Leyendo Prestamos_completo.csv...")
    pre = pd.read_csv("data/Prestamos_completo.csv", low_memory=False)

    col[COL_BARCODE] = col[COL_BARCODE].astype(str).str.strip()
    col[COL_INSTANCE_ID] = col[COL_INSTANCE_ID].astype(str)
    for c in [COL_TITLE, COL_SUBJECTS, COL_PUBLICATION, COL_CALL_NUMBER, COL_MTYPE, COL_LOCATION]:
        col[c] = col[c].fillna("").astype(str)
    col[COL_CONTRIBUTORS] = col[COL_CONTRIBUTORS].fillna("").astype(str)
    print("Parseando contribuidores/publicación (JSON por fila)...")
    col["contributors_text"] = col[COL_CONTRIBUTORS].apply(contrib)
    col["publication_text"] = col[COL_PUBLICATION].apply(publication_text)

    print("Agrupando a nivel de obra...")
    obras = (col.groupby(COL_INSTANCE_ID)
             .agg({COL_TITLE: "first", COL_SUBJECTS: ju, "publication_text": ju, "contributors_text": ju,
                   COL_CALL_NUMBER: "first", COL_MTYPE: "first", COL_LOCATION: "first"})
             .reset_index().rename(columns={COL_INSTANCE_ID: "instance_id"}))
    b2i = (col[[COL_BARCODE, COL_INSTANCE_ID]].dropna().drop_duplicates()
           .set_index(COL_BARCODE)[COL_INSTANCE_ID].to_dict())

    sw_es = ["de","la","el","los","las","y","en","del","a","por","para","con","una","un","al","se","su","sus","como","mas","o","e","que","es","sobre","entre","sin","edicion","vol","ed"]
    sw_pt = ["de","da","do","das","dos","e","em","um","uma","para","com","por","que","se","na","no","nas","nos","ao","aos"]
    sw_fr = ["de","la","le","les","et","en","du","des","un","une","par","sur","dans","avec","pour","au","aux"]
    sw = list(set(sw_es + sw_pt + sw_fr + list(ENGLISH_STOP_WORDS)))
    obras["subjects_clean"] = obras[COL_SUBJECTS].str.replace(";", " ", regex=False)
    obras["content_text"] = (obras[COL_TITLE] + " " + obras["subjects_clean"] + " " + obras["subjects_clean"] + " " +
                              obras["contributors_text"] + " " + obras["publication_text"] + " " +
                              obras[COL_MTYPE] + " " + obras[COL_LOCATION]).apply(nt)

    print("Ajustando TF-IDF (50000 features, bigramas)...")
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, max_df=0.85, stop_words=sw)
    X = vec.fit_transform(obras["content_text"]).astype(np.float32)
    print(f"  vocabulario: {len(vec.vocabulary_)}, matriz: {X.shape}, nnz: {X.nnz:,}")

    pre[LOAN_USER] = pre[LOAN_USER].astype(str).str.strip()
    pre[LOAN_BARCODE] = pre[LOAN_BARCODE].astype(str).str.strip()
    pre[LOAN_DATE] = pd.to_datetime(pre[LOAN_DATE], errors="coerce")
    pre = pre.dropna(subset=[LOAN_DATE])
    pre = pre[pre[LOAN_ACTION].isin(ACCIONES_VALIDAS)].copy()
    pre["instance_id"] = pre[LOAN_BARCODE].map(b2i)
    pre = pre.dropna(subset=["instance_id"]).copy()
    mf = pre[LOAN_DATE].max()
    pre["dias"] = (mf - pre[LOAN_DATE]).dt.days.fillna(730)
    pre["peso_recencia"] = np.exp(-pre["dias"] / 180)
    pre["peso_base"] = np.where(pre[LOAN_ACTION].str.contains("Renewed", case=False), 0.5, 1.0)
    pre["peso_evento"] = pre["peso_base"] * (1 + pre["peso_recencia"])
    pre[LOAN_FACULTY] = pre[LOAN_FACULTY].fillna("").astype(str)
    pre[LOAN_PROFILE] = pre[LOAN_PROFILE].fillna("").astype(str)
    pre[LOAN_PROGRAM] = pre[LOAN_PROGRAM].fillna("").astype(str)

    print("Agregando interacciones usuario-obra...")
    inter = (pre.rename(columns={LOAN_USER: "user_id", LOAN_BARCODE: "barcode", LOAN_FACULTY: "facultad",
                                  LOAN_PROFILE: "perfil_prestamo", LOAN_PROGRAM: "programa_prestamo"})
             .groupby(["user_id", "instance_id"])
             .agg(peso=("peso_evento", "sum"), n=("barcode", "count"), facultad=("facultad", "first"),
                  perfil_prestamo=("perfil_prestamo", "first"), programa_prestamo=("programa_prestamo", "first"))
             .reset_index())
    inter["peso"] = np.log1p(inter["peso"])
    inter["facultad_clean"] = inter["facultad"].map(MAPEO_FAC).fillna("")
    inter["programa_clean"] = inter["programa_prestamo"].apply(nt)

    pop_fac = (inter.groupby(["facultad_clean", "instance_id"])
               .agg(n_usuarios=("user_id", "nunique")).reset_index())
    pop_fac["score_pop"] = np.log1p(pop_fac["n_usuarios"])
    pop_gen = (inter.groupby("instance_id")["user_id"].nunique()
               .reset_index(name="n").sort_values("n", ascending=False))

    pop_prog = (inter[inter["perfil_prestamo"] == PERFIL_ESTUDIANTE]
                .groupby(["programa_clean", "instance_id"])
                .agg(n_usuarios=("user_id", "nunique")).reset_index())
    pop_prog["score_pop"] = np.log1p(pop_prog["n_usuarios"])

    print(f"Guardando artefactos en {out_dir}/ ...")
    obras.to_parquet(f"{out_dir}/obras.parquet", index=False)
    inter.to_parquet(f"{out_dir}/inter.parquet", index=False)
    pop_fac.to_parquet(f"{out_dir}/pop_fac.parquet", index=False)
    pop_prog.to_parquet(f"{out_dir}/pop_prog.parquet", index=False)
    pop_gen.to_parquet(f"{out_dir}/pop_gen.parquet", index=False)
    sparse.save_npz(f"{out_dir}/tfidf_matrix.npz", X)
    import joblib
    joblib.dump(vec, f"{out_dir}/tfidf_vectorizer.joblib", compress=3)

    import os as _os
    for fn in ["obras.parquet","inter.parquet","pop_fac.parquet","pop_prog.parquet","pop_gen.parquet","tfidf_matrix.npz","tfidf_vectorizer.joblib"]:
        size_mb = _os.path.getsize(f"{out_dir}/{fn}") / 1024 / 1024
        print(f"  {fn}: {size_mb:.1f} MB")

    print(f"Listo en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
