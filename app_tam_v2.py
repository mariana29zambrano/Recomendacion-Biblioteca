import streamlit as st
import pandas as pd
import numpy as np
import re
import base64
from datetime import datetime
from unidecode import unidecode
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Sistema de Recomendación — Biblioteca Javeriana",
    page_icon="📚", layout="wide", initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* La app está diseñada solo para tema claro (ver .streamlit/config.toml).
   color-scheme:light evita que navegadores con "forzar modo oscuro" (común en
   Android) reinterpreten los colores y vuelvan invisible el texto sin estilo
   propio (títulos, labels de campos, radios, expanders) sobre nuestros fondos
   claros. */
:root { color-scheme: light; }

/* Nunca permitir scroll horizontal de la página: nada debe poder empujar el ancho
   más allá del viewport (red de seguridad además del max-width de abajo). */
html, body { overflow-x: hidden; max-width: 100vw; }

/* Ocultar elementos de Streamlit */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Contenedor real del contenido (Streamlit no permite envolverlo en un div propio
   entre llamadas a st.markdown, ver CLAUDE.md): el límite de ancho y el centrado
   se aplican acá directamente, no en .page-inner (que quedó sin usar). */
section[data-testid="stMain"] > div {
    padding: 40px !important;
    max-width: 940px; margin: 0 auto !important;
    width: 100%; overflow-x: hidden;
}
section[data-testid="stMain"] > div * { max-width: 100%; }

/* Sidebar izquierda */
.app-sidebar {
    width: 220px; min-width: 220px;
    background: #002147;
    display: flex; flex-direction: column;
    align-items: center; padding: 32px 16px;
    position: fixed; top: 0; left: 0; height: 100vh;
    z-index: 100;
}
.app-sidebar .logo-container {
    display: flex; flex-direction: column;
    align-items: center; width: 100%;
}
.app-sidebar .logo-img {
    width: 100%; max-width: 180px; height: auto;
}
.sidebar-footer {
    margin-top: auto; color: rgba(255,255,255,0.5);
    font-size: 10px; text-align: center; line-height: 1.5;
}

/* Contenido principal */
section[data-testid="stMain"] {
    margin-left: 220px; min-height: 100vh;
    max-width: calc(100vw - 220px);
    background: #F8FAFC;
}

/* Pantallas angostas (laptops pequeños/tablet): el sidebar fijo de 220px deja de
   valer la pena, se colapsa para que el contenido tenga todo el ancho disponible. */
@media (max-width: 900px) {
    .app-sidebar { display: none; }
    section[data-testid="stMain"] { margin-left: 0; max-width: 100vw; }
    section[data-testid="stMain"] > div { padding: 24px !important; }
}

/* Progress bar */
.progress-wrap {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin-bottom: 36px;
}
.prog-step {
    display: flex; flex-direction: column; align-items: center;
    min-width: 80px;
}
.prog-circle {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; border: 2px solid #CBD5E1;
    background: white; color: #94A3B8;
}
.prog-circle.done { background: #002147; color: white; border-color: #002147; }
.prog-circle.active { background: #002147; color: white; border-color: #002147; }
.prog-label { font-size: 11px; color: #94A3B8; margin-top: 5px; font-weight: 500; }
.prog-label.active { color: #002147; font-weight: 700; }
.prog-label.done { color: #002147; }
.prog-line { width: 60px; height: 2px; background: #E2E8F0; margin-bottom: 20px; }
.prog-line.done { background: #002147; }

/* Celular (la mayoría de participantes del estudio usan el teléfono): apilar
   TODOS los layouts de columnas de Streamlit a ancho completo en vez de
   encoger N columnas en ~350px — aplica por igual al formulario de 2 columnas,
   a las 7 columnas de la escala Likert del TAM y a los pares de botones. */
@media (max-width: 600px) {
    section[data-testid="stMain"] > div { padding: 16px !important; }
    div[data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    div[data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
    div[data-testid="stElementContainer"] { width: 100% !important; }
    div[data-testid="stButton"] { width: 100% !important; }
    div[data-testid="stButton"] > button { width: 100% !important; padding: 12px 16px !important; min-height: 44px; }
    .progress-wrap { gap: 0; margin-bottom: 24px; }
    .prog-step { min-width: 48px; }
    .prog-circle { width: 26px; height: 26px; font-size: 11px; }
    .prog-label { font-size: 9px; }
    .prog-line { width: 20px; margin-bottom: 16px; }
}

/* Input fields */
.stTextInput > div > div > input {
    border: 1.5px solid #E2E8F0 !important; border-radius: 8px !important;
    font-size: 13px !important; padding: 10px 14px !important;
}
.stSelectbox > div > div {
    border: 1.5px solid #E2E8F0 !important; border-radius: 8px !important;
}

/* Botones de Streamlit */
div[data-testid="stButton"] > button {
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 14px !important; padding: 10px 28px !important;
    border: 1.5px solid #E2E8F0 !important; background: white !important;
    color: #374151 !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: #002147 !important; color: white !important;
    border-color: #002147 !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #002147 !important; color: #002147 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #001529 !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
SHEET_ID   = "1TxN5DbrjMhGoMaLQOvSXB2uGuwLDSBbOfVHyvJ3Bv1Q"
SHEET_NAME = "Hoja 1"
TOP_N      = 10

COL_TITLE       = "Instances - Index title"
COL_SUBJECTS    = "Instances - Subject headings"
COL_CALL_NUMBER = "Items - Item call number"
COL_MTYPE       = "Material type - Name"

# Pesos del score híbrido del Modelo 1 (óptimo empírico de la sección 10.1 del
# notebook, gs_pesos_m1_heatmap.png: mejor Recall@10 y NDCG@10 simultáneamente
# con w_pop=0.4 fijo y w_hist=0.4/w_perf=0.2 sobre el remanente). Usado por
# rec_A (flujo manual, datos de la iteración 2 -- inter/pop_fac/pop_prog).
W_HIST = 0.40
W_PERF = 0.20
W_POP  = 0.40

# Pesos recalibrados para rec_A_v3 (tercera iteración: directorio institucional
# completo + histórico extendido, ya filtrado a Bogotá -- ver
# scripts/grid_search_pesos_v3.py y gs_pesos_m1_v3_heatmap.png). No son los
# mismos de arriba a propósito: se calibraron sobre una matriz de interacciones
# mucho más densa (score_historial real para ~97% de los participantes en vez
# de casi siempre 0), así que el punto óptimo es distinto -- Burke (1999) es
# explícito en que los pesos de un híbrido deben calibrarse empíricamente por
# dataset, no reusarse de una iteración con datos muy distintos.
W_HIST_V3 = 0.50
W_PERF_V3 = 0.30
W_POP_V3  = 0.20

FACULTADES_PROGRAMAS = {
    "Facultad de Arquitectura y Diseño":["Arquitectura","Diseño Industrial","Maestría en Diseño para la Innovación de Productos y Servicios","Maestría en Hábitat Sustentable","Maestría en Patrimonio Cultural y Territorio","Maestría en Planeación Urbana y Regional","Especialización en Gerencia de Proyectos de Diseño"],
    "Facultad de Artes":["Artes Escénicas","Artes Visuales","Estudios Musicales","Maestría en Creación Audiovisual","Maestría en Música"],
    "Facultad de Ciencias":["Bacteriología","Biología","Ciencia de Datos","Matemáticas","Microbiología Agrícola y Veterinaria","Microbiología Industrial","Nutrición y Dietética","Química Farmacéutica","Doctorado en Ciencias Biológicas","Maestría en Ciencias Biológicas","Maestría en Ciencias del Laboratorio Clínico","Maestría en Física Médica","Maestría en Matemáticas","Maestría en Microbiología","Maestría en Restauración Ecológica","Especialización en Análisis Químico Instrumental","Especialización en Microbiología Médica"],
    "Facultad de Ciencias Económicas y Administrativas":["Administración de Empresas","Contaduría Pública","Economía","Finanzas","Negocios Internacionales","Doctorado en Economía","Maestría en Administración","Maestría en Administración de Salud","Maestría en Banca y Finanzas","Maestría en Economía","Maestría en Finanzas Aplicadas","Maestría en Gerencia de la Sostenibilidad","Maestría en Estrategia, Innovación y Competitividad","Especialización en Gerencia Financiera","Especialización en Marketing Estratégico"],
    "Facultad de Ciencias Jurídicas":["Derecho","Doctorado en Ciencias Jurídicas","Maestría en Derecho Administrativo","Maestría en Derecho Constitucional","Maestría en Derecho Económico","Maestría en Derecho Laboral y de la Seguridad Social","Especialización en Derecho Administrativo","Especialización en Derecho Comercial","Especialización en Derecho de Familia","Especialización en Derecho Laboral","Especialización en Derecho Tributario"],
    "Facultad de Ciencias Políticas y Relaciones Internacionales":["Ciencia Política","Relaciones Internacionales","Maestría en Estudios Contemporáneos de América Latina","Maestría en Estudios de Paz y Resolución de Conflictos","Maestría en Estudios Internacionales","Maestría en Estudios Políticos","Maestría en Gobierno del Territorio y Gestión Pública","Maestría en Política Social","Especialización en Gobierno Municipal","Especialización en Resolución de Conflictos"],
    "Facultad de Ciencias Sociales":["Antropología","Historia","Sociología","Estudios Literarios","Doctorado en Ciencias Sociales y Humanas","Maestría en Estudios Afrocolombianos","Maestría en Estudios Culturales","Maestría en Historia","Maestría en Literatura","Especialización en Literatura Infantil y Juvenil"],
    "Facultad de Comunicación y Lenguaje":["Ciencia de la Información, Bibliotecología y Archivística","Comunicación Social","Licenciatura en Lenguas Modernas con Énfasis en Inglés y Francés","Doctorado en Comunicación, Lenguajes e Información","Maestría en Archivística Histórica y Memoria","Maestría en Comunicación, Tecnología y Sociedad","Maestría en Periodismo Científico","Especialización en Comunicación Organizacional"],
    "Facultad de Derecho Canónico":["Licenciatura Eclesiástica en Derecho Canónico","Doctorado Eclesiástico en Derecho Canónico","Maestría en Derecho Canónico","Especialización en Derecho Matrimonial Canónico"],
    "Facultad de Educación":["Licenciatura en Ciencias Naturales y Educación Ambiental","Licenciatura en Educación Básica con Énfasis en Humanidades y Lengua Castellana","Licenciatura en Educación Básica Primaria","Licenciatura en Educación Física","Licenciatura en Educación Infantil","Licenciatura en Filosofía","Licenciatura en Literatura y Lengua Castellana","Maestría en Educación","Maestría en Investigación y Tecnología Educativa","Especialización en Liderazgo para la Gestión Social"],
    "Facultad de Enfermería":["Enfermería","Maestría en Enfermería en Cuidado Crítico","Maestría en Enfermería en Cuidado Paliativo","Maestría en Seguridad y Salud en el Trabajo","Especialización en Enfermería en Cuidado Crítico","Especialización en Enfermería Pediátrica"],
    "Facultad de Estudios Ambientales y Rurales":["Ecología","Doctorado en Estudios Ambientales y Rurales","Maestría en Conservación y Uso de Biodiversidad","Maestría en Desarrollo Rural","Maestría en Gestión Ambiental","Maestría en Saneamiento y Desarrollo Ambiental"],
    "Facultad de Filosofía":["Filosofía","Licenciatura en Filosofía","Doctorado en Filosofía","Maestría en Bioética","Maestría en Filosofía","Especialización en Bioética"],
    "Facultad de Ingeniería":["Bioingeniería","Ingeniería Civil","Ingeniería de Sistemas","Ingeniería Electrónica","Ingeniería en Redes y Telecomunicaciones","Ingeniería Industrial","Ingeniería Mecánica","Ingeniería Mecatrónica","Doctorado en Ingeniería","Maestría en Analítica para la Inteligencia de Negocios","Maestría en Ingeniería Civil","Maestría en Ingeniería de Sistemas y Computación","Maestría en Ingeniería Electrónica","Maestría en Ingeniería Industrial","Maestría en Inteligencia Artificial","Maestría en Logística y Transporte","Maestría en Seguridad Digital","Especialización en Arquitectura Empresarial de Software","Especialización en Inteligencia Artificial"],
    "Facultad de Medicina":["Medicina","Doctorado en Epidemiología Clínica","Doctorado en Neurociencias","Maestría en Bioestadística","Maestría en Epidemiología Clínica","Especialización en Anestesiología","Especialización en Cirugía General","Especialización en Ginecología y Obstetricia","Especialización en Medicina Familiar","Especialización en Medicina Interna","Especialización en Neurología","Especialización en Pediatría","Especialización en Psiquiatría"],
    "Facultad de Odontología":["Odontología","Especialización en Cirugía Maxilofacial","Especialización en Endodoncia","Especialización en Odontopediatría","Especialización en Ortodoncia","Especialización en Periodoncia","Especialización en Rehabilitación Oral"],
    "Facultad de Psicología":["Psicología","Doctorado en Psicología","Maestría en Abordajes Psicosociales para la Construcción de Culturas de Paz","Maestría en Psicología Clínica","Maestría en Psicología Comunitaria"],
    "Facultad de Teología":["Teología","Licenciatura en Ciencias Religiosas","Licenciatura en Teología","Doctorado en Teología","Maestría en Teología"],
}

PERFILES = ["Estudiante de pregrado","Estudiante de posgrado — Especialización","Estudiante de posgrado — Maestría","Estudiante de posgrado — Doctorado","Profesor/a","Personal administrativo","Otro"]
CARGOS   = ["Profesor titular","Profesor asociado","Profesor asistente","Instructor","Profesor de cátedra","Otro"]

# ── Google Sheets ──────────────────────────────────────────────────────────────
@st.cache_resource
def conectar_sheets():
    scopes=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    # En Streamlit Cloud las credenciales viven en st.secrets (no se puede subir
    # credentials.json al repo); en local, si no hay secrets.toml, cae al archivo
    # (st.secrets lanza StreamlitSecretNotFoundError si no existe ningún secrets.toml).
    try:
        tiene_secrets = "gcp_service_account" in st.secrets
    except Exception:
        tiene_secrets = False
    if tiene_secrets:
        creds=Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]),scopes=scopes)
    else:
        creds=Credentials.from_service_account_file("credentials.json",scopes=scopes)
    client=gspread.authorize(creds)
    sheet=client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
    if not sheet.get_all_values():
        # update() en vez de append_row(): escribe directo en la fila 1, evitando la
        # condicion de carrera donde append_row (header) y el primer guardar() (dato)
        # calculan "siguiente fila vacia" casi al tiempo y ambos aterrizan en la fila 1.
        sheet.update([["timestamp","edad","perfil","facultad","programa","semestre_cargo","genero","tiene_prestamos","titulo_recordado","algoritmo","favoritos","PU1","PU2","PU3","PU4","PEOU1","PEOU2","PEOU3","PEOU4","REL1","REL2","REL3","OUT1","OUT2","OUT3","BI1","BI2","BI3"]], "A1")
    return sheet

def guardar(sheet, d):
    sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        d.get("edad",""),d.get("perfil",""),d.get("facultad",""),d.get("programa",""),
        d.get("semestre_cargo",""),d.get("genero",""),d.get("tiene_prestamos",""),d.get("titulo_recordado",""),
        d.get("algoritmo",""),d.get("favoritos",""),
        d.get("PU1",""),d.get("PU2",""),d.get("PU3",""),d.get("PU4",""),
        d.get("PEOU1",""),d.get("PEOU2",""),d.get("PEOU3",""),d.get("PEOU4",""),
        d.get("REL1",""),d.get("REL2",""),d.get("REL3",""),
        d.get("OUT1",""),d.get("OUT2",""),d.get("OUT3",""),
        d.get("BI1",""),d.get("BI2",""),d.get("BI3",""),
    ])

def nt(t):
    """Normaliza texto (minúsculas, sin tildes, solo alfanumérico, espacios colapsados).
    Se usa tanto para content_text del TF-IDF como para las claves de matching de
    facultad/programa entre lo que escribe el usuario y lo que aparece en los préstamos."""
    t=str(t).lower(); t=unidecode(t); t=re.sub(r"[^a-z0-9\s]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

# IDs de Google Drive de los artefactos PRECOMPUTADOS (ver scripts/precompute_cache.py
# y scripts/subir_cache_drive.py). Antes la app descargaba los CSV crudos (470MB+64MB)
# y los reprocesaba —parseo JSON fila por fila, TF-IDF de 329k obras— en cada arranque
# en frío: varios minutos y picos de RAM que hacían caer el proceso en el plan gratuito
# de Streamlit Cloud (1GB) con varios usuarios simultáneos. Ahora se descargan los
# resultados ya calculados (Parquet + npz + joblib) y solo se cargan, sin reprocesar.
# Si cambian los CSV fuente, hay que correr esos dos scripts de nuevo y actualizar
# estos IDs.
ID_ARTEFACTOS = {
    "inter.parquet":            "1jYmvk4GNJE9fyhtUczZv9sejRdT9GmsA",
    "obras.parquet":            "1TZdB6cCAZuIJM_p1OoJqf_NAcmlFy_jT",
    "pop_fac.parquet":          "1oxdb8vco8PCIBtVuPo8RWKJ47K7GcJ34",
    "pop_gen.parquet":          "1smwjIGDzIj8_Q5qq_kfiWVCPyq3vRq9Y",
    "pop_prog.parquet":         "1yZLim_SvPyp5mgPq4F4x3mbCOKVXvxs9",
    "tfidf_matrix.npz":         "1FsyQTssDf6h6t6cL9MYdBG4k30idYUu0",
    "tfidf_vectorizer.joblib":  "13nR2Y6NT4LMe4v6mF39BfI2-EAFVDrnS",
}

# Artefactos de la TERCERA iteracion (scripts/precompute_cache_v3.py), generados
# localmente a partir del directorio institucional completo + historico extendido
# (tercera_iteracion/, nunca subido al repo). Si en el futuro cambia la fuente y
# hay que regenerarlos, subir los nuevos a Drive y actualizar los IDs de abajo --
# si alguno llegara a faltar, cargar_datos() sigue funcionando igual (usa lo que
# ya exista en cache/ localmente) y la app simplemente no ofrece el flujo por
# "ID de estudiante" hasta que estén todos.
ID_ARTEFACTOS_V3 = {
    "inter_v3.parquet":         "1mhBWHlufLAqvxasgC-qLBCUz64r3UaQ5",
    "perfil_v3.parquet":        "1sYKEjs9ZfH0_lvELCf7loiGB4kNE0yk_",
    "pop_fac_v3.parquet":       "1xLfpKjevyiGqS5cB7a-iGPEiHpMxvfAs",
    "pop_prog_v3.parquet":      "142QJdczlFAa2nk7_baU7GfrFB7gZnjag",
    "estudiante_lookup.parquet":"1ww4n8NEYN3Z0vOd-TOa2_S7eYDcbCcQP",
}

# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando la colección bibliográfica...")
def cargar_datos():
    import os, gdown, joblib
    from scipy import sparse

    os.makedirs("cache", exist_ok=True)
    for nombre, file_id in ID_ARTEFACTOS.items():
        ruta = f"cache/{nombre}"
        if not os.path.exists(ruta):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", ruta, quiet=False)
    for nombre, file_id in ID_ARTEFACTOS_V3.items():
        ruta = f"cache/{nombre}"
        if not os.path.exists(ruta) and file_id:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", ruta, quiet=False)

    obras   = pd.read_parquet("cache/obras.parquet")
    inter   = pd.read_parquet("cache/inter.parquet")
    pop_fac = pd.read_parquet("cache/pop_fac.parquet")
    pop_prog= pd.read_parquet("cache/pop_prog.parquet")
    pop_gen = pd.read_parquet("cache/pop_gen.parquet")
    X       = sparse.load_npz("cache/tfidf_matrix.npz")
    vec     = joblib.load("cache/tfidf_vectorizer.joblib")
    book_to_row=pd.Series(obras.index.values,index=obras["instance_id"]).to_dict()

    datos = {"obras":obras,"inter":inter,"pop_fac":pop_fac,"pop_prog":pop_prog,"pop_gen":pop_gen,
             "vec":vec,"X":X,"book_to_row":book_to_row}

    # Artefactos v3: opcionales -- si no estan disponibles (todavia no
    # desplegados), el flujo por "ID de estudiante" simplemente no se activa
    # y la app sigue funcionando con el flujo manual de siempre.
    v3_completo = all(os.path.exists(f"cache/{n}") for n in ID_ARTEFACTOS_V3)
    datos["v3_disponible"] = v3_completo
    if v3_completo:
        datos["inter_v3"]    = pd.read_parquet("cache/inter_v3.parquet")
        datos["perfil_v3"]   = pd.read_parquet("cache/perfil_v3.parquet").set_index("id")
        datos["pop_fac_v3"]  = pd.read_parquet("cache/pop_fac_v3.parquet")
        datos["pop_prog_v3"] = pd.read_parquet("cache/pop_prog_v3.parquet")
        datos["estudiante_lookup"] = pd.read_parquet("cache/estudiante_lookup.parquet")

    return datos

# ── Recomendaciones ────────────────────────────────────────────────────────────
def rec_A(fac,prog,datos,n=TOP_N,carnet="",perfil="",genero=""):
    """Modelo 1 (TF-IDF híbrido) — score = W_HIST*historial + W_PERF*perfil + W_POP*popularidad,
    pesos óptimos de la sección 10.1 del notebook (gs_pesos_m1_heatmap.png).
    score_perfil usa facultad+programa+perfil(rol)+género como proxy del profile_text real del
    notebook (construir_perfil): ahí se arma desde el historial de préstamos —
    perfil_prestamo+programa_prestamo+facultad_prestamo+género+tipo_usuario+programa_digital+
    facultad_digital+nivel_formación (los últimos 4 salen de "BD Digitales", que no está en este
    repo). Acá se usa lo que el participante escribe en el formulario en su lugar, ya que el
    Este es el fallback cuando no se encuentra un "ID de estudiante" real (ver
    buscar_id_interno/rec_A_v3 mas abajo): el formulario ya no pide carnet, asi que
    el parametro `carnet` queda en "" salvo que se llame manualmente -- score_historial
    siempre es cero aqui y el score se apoya en perfil + popularidad. Se conserva el
    parametro para no perder la replicacion fiel del Modelo 1 del notebook.
    La popularidad usa el programa (carrera) del usuario, filtrada a estudiantes; si ese
    programa no tiene préstamos registrados, cae a popularidad por facultad."""
    obras=datos["obras"]; pop_fac=datos["pop_fac"]; pop_prog=datos["pop_prog"]
    vec=datos["vec"]; X=datos["X"]
    inter=datos["inter"]; book_to_row=datos["book_to_row"]

    carnet=str(carnet).strip()
    historial=inter[inter["user_id"]==carnet] if carnet else inter.iloc[0:0]
    score_hist=np.zeros(X.shape[0])
    if not historial.empty:
        idx=historial["instance_id"].map(book_to_row).dropna().astype(int)
        pesos=historial.loc[idx.index,"peso"].values
        if len(idx)>0 and pesos.sum()>0:
            uv=X[idx.values].multiply(pesos.reshape(-1,1)).sum(axis=0)
            uv=csr_matrix(uv/pesos.sum())
            score_hist=cosine_similarity(uv,X).ravel()

    txt=re.sub(r"[^a-z0-9\s]"," ",unidecode(f"{fac} {prog} {perfil} {genero}".lower())).strip()
    score_perfil=np.zeros(X.shape[0])
    if txt: score_perfil=cosine_similarity(vec.transform([txt]),X).ravel()

    cand=obras.copy(); cand["score_historial"]=score_hist; cand["score_perfil"]=score_perfil
    if not historial.empty:
        cand=cand[~cand["instance_id"].isin(set(historial["instance_id"]))]

    pf=pop_fac[pop_fac["facultad_clean"]==fac][["instance_id","score_pop"]].rename(columns={"score_pop":"score_pop_fac"})
    cand=cand.merge(pf,on="instance_id",how="left"); cand["score_pop_fac"]=cand["score_pop_fac"].fillna(0)

    prog_clean=nt(prog)
    pp=pop_prog[pop_prog["programa_clean"]==prog_clean][["instance_id","score_pop"]].rename(columns={"score_pop":"score_pop_prog"})
    cand=cand.merge(pp,on="instance_id",how="left"); cand["score_pop_prog"]=cand["score_pop_prog"].fillna(0)

    usa_programa=cand["score_pop_prog"].sum()>0
    score_pop_base=cand["score_pop_prog"] if usa_programa else cand["score_pop_fac"]
    mp=score_pop_base.max(); cand["score_pop_norm"]=score_pop_base/mp if mp>0 else 0

    cand["score"]=W_HIST*cand["score_historial"]+W_PERF*cand["score_perfil"]+W_POP*cand["score_pop_norm"]
    # El catalogo puede tener mas de un instance_id para la misma obra (ej. DVD vs
    # Blu-ray de la misma pelicula, mismo content_text): sin este dedup, ambos quedan
    # con score casi identico y se repite el titulo. Clave = titulo+contribuidores,
    # NO solo titulo: 12,922 titulos del catalogo (ej. "Memorias", "Obras completas")
    # agrupan decenas/cientos de obras DISTINTAS con el mismo titulo generico --
    # deduplicar solo por titulo las colapsaria en una sola recomendacion incorrecta.
    cand["_dup_key"]=cand[COL_TITLE].map(nt)+"||"+cand["contributors_text"].fillna("").map(nt)
    cand=cand.sort_values("score",ascending=False).drop_duplicates("_dup_key",keep="first")
    return cand.head(n)[["instance_id",COL_TITLE,COL_SUBJECTS,COL_MTYPE,COL_CALL_NUMBER,"contributors_text"]].reset_index(drop=True)

def rec_C(titulo,datos,n=TOP_N):
    obras=datos["obras"]; vec=datos["vec"]; X=datos["X"]
    if not titulo or not titulo.strip():
        pg=datos["pop_gen"].head(n)
        return obras[obras["instance_id"].isin(pg["instance_id"])][["instance_id",COL_TITLE,COL_SUBJECTS,COL_MTYPE,COL_CALL_NUMBER,"contributors_text"]].reset_index(drop=True)
    txt=re.sub(r"[^a-z0-9\s]"," ",unidecode(titulo.lower())).strip()
    sims=cosine_similarity(vec.transform([txt]),X).ravel()
    cand=obras.copy(); cand["score"]=sims
    cand["_dup_key"]=cand[COL_TITLE].map(nt)+"||"+cand["contributors_text"].fillna("").map(nt)
    cand=cand.sort_values("score",ascending=False).drop_duplicates("_dup_key",keep="first")
    return cand.iloc[1:n+1][["instance_id",COL_TITLE,COL_SUBJECTS,COL_MTYPE,COL_CALL_NUMBER,"contributors_text"]].reset_index(drop=True)

def buscar_id_interno(id_estudiante, datos):
    """Traduce el 'ID de estudiante' que escribe el participante al id interno
    (UUID) usando cache/estudiante_lookup.parquet (solo dos columnas: id_estudiante,
    id -- ver scripts/precompute_cache_v3.py). Devuelve None si no hay match o si
    los artefactos v3 no estan disponibles; el id_estudiante NUNCA se guarda en
    session_state mas alla de esta busqueda puntual ni se persiste en la hoja."""
    if not datos.get("v3_disponible") or not id_estudiante or not str(id_estudiante).strip():
        return None
    lookup = datos["estudiante_lookup"]
    fila = lookup[lookup["id_estudiante"] == str(id_estudiante).strip()]
    return fila["id"].iloc[0] if not fila.empty else None

def rec_A_v3(id_interno, datos, n=TOP_N):
    """Equivalente de rec_A pero con datos reales de la tercera iteracion: historial
    real de prestamos (inter_v3), perfil institucional real en vez de lo que el
    participante escribe a mano (perfil_v3: facultad/programa normalizados +
    tipo de usuario), y popularidad ya escopada a Bogota (pop_fac_v3/pop_prog_v3).
    Pesos W_HIST_V3/W_PERF_V3/W_POP_V3 (0.50/0.30/0.20), recalibrados con
    scripts/grid_search_pesos_v3.py sobre esta matriz de interacciones mucho mas
    densa que la de la iteracion 2 -- NDCG@10=0.0368, mejor que historial solo
    (0.0264) o popularidad sola (0.0240). Ver gs_pesos_m1_v3_heatmap.png."""
    obras=datos["obras"]; vec=datos["vec"]; X=datos["X"]; book_to_row=datos["book_to_row"]
    inter_v3=datos["inter_v3"]; perfil_v3=datos["perfil_v3"]
    pop_fac_v3=datos["pop_fac_v3"]; pop_prog_v3=datos["pop_prog_v3"]

    historial=inter_v3[inter_v3["id"]==id_interno]
    score_hist=np.zeros(X.shape[0])
    if not historial.empty:
        idx=historial["instance_id"].map(book_to_row).dropna().astype(int)
        pesos=historial.loc[idx.index,"peso"].values
        if len(idx)>0 and pesos.sum()>0:
            uv=X[idx.values].multiply(pesos.reshape(-1,1)).sum(axis=0)
            uv=csr_matrix(uv/pesos.sum())
            score_hist=cosine_similarity(uv,X).ravel()

    fila_perfil = perfil_v3.loc[id_interno] if id_interno in perfil_v3.index else None
    facultad_usuario = fila_perfil["facultad_clean"] if fila_perfil is not None else ""
    programa_usuario = fila_perfil["programa_clean"] if fila_perfil is not None else ""
    tipo_usuario = fila_perfil["tipo_usuario"] if fila_perfil is not None else ""

    txt = f"{facultad_usuario} {programa_usuario} {tipo_usuario}".strip()
    score_perfil=np.zeros(X.shape[0])
    if txt: score_perfil=cosine_similarity(vec.transform([txt]),X).ravel()

    cand=obras.copy(); cand["score_historial"]=score_hist; cand["score_perfil"]=score_perfil
    if not historial.empty:
        cand=cand[~cand["instance_id"].isin(set(historial["instance_id"]))]

    pf=pop_fac_v3[pop_fac_v3["facultad_clean"]==facultad_usuario][["instance_id","score_pop"]].rename(columns={"score_pop":"score_pop_fac"})
    cand=cand.merge(pf,on="instance_id",how="left"); cand["score_pop_fac"]=cand["score_pop_fac"].fillna(0)

    pp=pop_prog_v3[pop_prog_v3["programa_clean"]==programa_usuario][["instance_id","score_pop"]].rename(columns={"score_pop":"score_pop_prog"})
    cand=cand.merge(pp,on="instance_id",how="left"); cand["score_pop_prog"]=cand["score_pop_prog"].fillna(0)

    usa_programa=cand["score_pop_prog"].sum()>0
    score_pop_base=cand["score_pop_prog"] if usa_programa else cand["score_pop_fac"]
    mp=score_pop_base.max(); cand["score_pop_norm"]=score_pop_base/mp if mp>0 else 0

    cand["score"]=W_HIST_V3*cand["score_historial"]+W_PERF_V3*cand["score_perfil"]+W_POP_V3*cand["score_pop_norm"]
    cand["_dup_key"]=cand[COL_TITLE].map(nt)+"||"+cand["contributors_text"].fillna("").map(nt)
    cand=cand.sort_values("score",ascending=False).drop_duplicates("_dup_key",keep="first")
    return cand.head(n)[["instance_id",COL_TITLE,COL_SUBJECTS,COL_MTYPE,COL_CALL_NUMBER,"contributors_text"]].reset_index(drop=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
def _logo_puj_b64():
    with open("Logo PUJ.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

def render_sidebar():
    st.markdown(f"""
    <div class="app-sidebar">
        <div class="logo-container">
            <img src="data:image/png;base64,{_logo_puj_b64()}" class="logo-img" alt="Pontificia Universidad Javeriana">
        </div>
        <div class="sidebar-footer">
            Biblioteca General<br>Alfonso Borrero Cabal S.J.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Progress bar ───────────────────────────────────────────────────────────────
def render_progress(paso):
    pasos = [("1","Perfil"),("2","Preferencias"),("3","Cuestionario"),("4","¡Gracias!")]
    html = '<div class="progress-wrap">'
    for i,(num,label) in enumerate(pasos):
        n = i+1
        if n < paso: cls="done"; icon="✓"
        elif n == paso: cls="active"; icon=num
        else: cls=""; icon=num
        html += f'<div class="prog-step"><div class="prog-circle {cls}">{icon}</div><div class="prog-label {cls}">{label}</div></div>'
        if i < len(pasos)-1:
            lc = "done" if n < paso else ""
            html += f'<div class="prog-line {lc}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ── Lista de libros ────────────────────────────────────────────────────────────
def render_libros(df, prefix):
    """Muestra libros con corazón vacío por defecto, relleno si se marca."""
    if df is None or df.empty:
        st.info("Sin recomendaciones disponibles.")
        return []
    favs = []
    for i, row in df.iterrows():
        key = f"{prefix}_{i}"
        if key not in st.session_state:
            st.session_state[key] = False
        liked = st.session_state[key]
        titulo = str(row[COL_TITLE])
        autor  = str(row.get("contributors_text","")).split(";")[0].strip()
        if not autor or autor == "nan": autor = ""
        corazon = "❤️" if liked else "♡"
        with st.expander(f"{corazon}  {titulo[:65]}{'...' if len(titulo)>65 else ''}", expanded=False):
            if autor:
                st.markdown(f"**Autor:** {autor}")
            mat = str(row[COL_SUBJECTS])
            st.markdown(f"**Materias:** {mat[:120]}..." if len(mat)>120 else f"**Materias:** {mat}")
            st.markdown(f"**Tipo:** {row[COL_MTYPE]} &nbsp;|&nbsp; **Signatura:** {row[COL_CALL_NUMBER]}")
            if st.button("Me gusta" if not liked else "Quitar", key=f"btn_{key}"):
                st.session_state[key] = not liked
                st.rerun()
        if st.session_state[key]:
            favs.append(titulo[:60])
    return favs

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    for k,v in [("pantalla","consentimiento"),("du",{}),("recs",{})]:
        if k not in st.session_state: st.session_state[k]=v

    pantalla = st.session_state.pantalla
    render_sidebar()

    # Contenedor principal (el margen/fondo del panel se aplican por CSS a section[data-testid="stMain"])
    with st.container():
        # ── CONSENTIMIENTO ─────────────────────────────────────────────────────
        if pantalla == "consentimiento":
            col_txt, col_img = st.columns([1.3, 1])
            with col_txt:
                st.markdown("# Sistema de Recomendación Bibliográfica")
                st.markdown("**Biblioteca General Alfonso Borrero Cabal S.J.**")
                st.info("Este es un proyecto desarrollado por el programa de Ciencia de Datos de la Pontificia Universidad Javeriana — Bogotá.")
                st.markdown("### Consentimiento informado para participación en prueba de usuario")
                st.markdown("""
Este ejercicio hace parte del trabajo de grado *"Diseño de un sistema de recomendación bibliográfica para la Biblioteca General Alfonso Borrero Cabal S.J."*, desarrollado en el programa de Ciencia de Datos de la Pontificia Universidad Javeriana.

**¿Qué harás?**
- Ingresarás información sobre tu perfil académico
- Recibirás una lista de recomendaciones de libros generada por el sistema
- Evaluarás su relevancia y responderás un breve cuestionario

**Sobre tus datos:**
- Tu participación es **voluntaria**
- Los datos recopilados se usarán **únicamente con fines académicos**
- No se compartirán con terceros ni se publicarán datos individuales
- Para personalizar tus recomendaciones, te pediremos tu ID de estudiante (obligatorio). Esta información se usa para consultar tu historial real de préstamos en la biblioteca y se usará **únicamente para este trabajo de grado**; no se comparte con terceros ni se publica de forma individual.
- Los datos podrán utilizarse en futuras iteraciones del proyecto

Los datos de este formulario serán usados exclusivamente para este trabajo de grado. Puede consultar la Política de Protección de Datos Personales de la Universidad en la página web www.javeriana.edu.co. El canal de comunicación para revocar la autorización otorgada o solicitar la supresión de los datos es el correo electrónico: usodedatos@javeriana.edu.co

*Al continuar autorizas el uso de la información ingresada bajo las condiciones descritas.*
                """)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Acepto y deseo participar", type="primary", use_container_width=True):
                        st.session_state.pantalla = "perfil"; st.rerun()
                with c2:
                    if st.button("No deseo participar", use_container_width=True):
                        st.info("Gracias. Puedes cerrar esta ventana.")

        # ── PERFIL (paso 1: ID + datos básicos para todos) ───────────────────────
        elif pantalla == "perfil":
            render_progress(1)
            st.markdown("## Cuéntanos sobre ti")
            st.caption("Esta información nos permite personalizar tus recomendaciones y mejorar el sistema en el futuro.")

            st.markdown("**Datos de identificación**")
            id_estudiante = st.text_input("ID de estudiante:", placeholder="Ej: 20221234567")
            edad   = st.text_input("Edad:", placeholder="Ej: 22")
            genero = st.selectbox("Género:", ["Prefiero no decirlo","Femenino","Masculino","No binario","Otro"])

            c1, c2 = st.columns([1,4])
            with c2:
                if st.button("Continuar", type="primary"):
                    if not id_estudiante.strip():
                        st.error("Por favor ingresa tu ID de estudiante.")
                    else:
                        with st.spinner("Verificando tu información..."):
                            datos = cargar_datos()
                            # El "ID de estudiante" se usa UNICAMENTE aqui, para
                            # resolver el id interno -- nunca se guarda en
                            # session_state.du ni se persiste en la hoja (ver
                            # buscar_id_interno() y la decision de proteccion de
                            # datos documentada en scripts/precompute_cache_v3.py).
                            id_interno = buscar_id_interno(id_estudiante, datos)
                            st.session_state.du = {"edad":edad,"genero":genero}
                            if id_interno is not None:
                                # Match real: ya tenemos historial + perfil institucional
                                # reales (rec_A_v3), no hace falta pedir el resto del
                                # formulario -- solo se usaría para segmentar el TAM, y
                                # eso ya no aplica aquí (decisión del 2026-10-08: no pedir
                                # datos de más si no se van a usar para nada).
                                st.session_state.du["algoritmo"] = "A_v3"
                                st.session_state.recs = {"lista": rec_A_v3(id_interno, datos)}
                                st.session_state.pantalla = "recomendaciones"
                            else:
                                st.session_state.pantalla = "perfil2"
                        st.rerun()

        # ── PERFIL (paso 2: resto de datos, solo si no hubo match por ID) ───────
        elif pantalla == "perfil2":
            render_progress(1)
            st.markdown("## Cuéntanos un poco más")
            st.caption("No encontramos tu ID en el directorio institucional — con estos datos generamos tu recomendación.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Perfil académico**")
                perfil   = st.selectbox("¿Cuál es tu rol en la universidad?", PERFILES)
                facultad = st.selectbox("Facultad:", ["— Selecciona —"]+sorted(FACULTADES_PROGRAMAS.keys()))
                programa = "No especificado"
                if facultad != "— Selecciona —":
                    programa = st.selectbox("Programa / Carrera:", FACULTADES_PROGRAMAS[facultad])
                semestre_cargo = ""
                if "Estudiante" in perfil:
                    semestre_cargo = st.selectbox("Semestre actual:", [str(s) for s in range(1,11)]+["11 o más"])
                elif perfil == "Profesor/a":
                    semestre_cargo = st.selectbox("Cargo:", CARGOS)

            with col2:
                st.markdown("**Historial en la biblioteca**")
                tiene = st.radio("¿Has realizado préstamos de libros en la Biblioteca General?", ["Sí","No"])
                titulo_rec = ""
                if tiene == "Sí":
                    titulo_rec = st.text_input("¿Recuerdas el título de algún libro que hayas prestado recientemente?", placeholder="Ej: Introducción a la estadística")

            c1, c2 = st.columns([1,4])
            with c2:
                if st.button("Continuar", type="primary"):
                    errores = []
                    if facultad == "— Selecciona —": errores.append("Por favor selecciona tu facultad.")
                    if ("Estudiante" in perfil or perfil == "Profesor/a") and not semestre_cargo: errores.append("Por favor selecciona tu semestre o cargo.")
                    if tiene == "Sí" and not titulo_rec.strip(): errores.append("Por favor indica el título del libro que has prestado.")
                    if errores:
                        for e in errores: st.error(e)
                    else:
                        st.session_state.du.update({"perfil":perfil,"facultad":facultad,"programa":programa,"semestre_cargo":semestre_cargo,"tiene_prestamos":tiene,"titulo_recordado":titulo_rec})
                        with st.spinner("Generando tus recomendaciones..."):
                            datos = cargar_datos()
                            du = st.session_state.du
                            if tiene == "Sí":
                                algoritmo = "C"
                                lista = rec_C(titulo_rec, datos)
                            else:
                                algoritmo = "A"
                                lista = rec_A(facultad, programa, datos, perfil=perfil, genero=du.get("genero",""))
                            st.session_state.du["algoritmo"] = algoritmo
                            st.session_state.recs = {"lista":lista}
                        st.session_state.pantalla = "recomendaciones"; st.rerun()

        # ── RECOMENDACIONES ────────────────────────────────────────────────────
        elif pantalla == "recomendaciones":
            render_progress(2)
            du = st.session_state.du
            st.markdown("## Recomendaciones para ti")
            st.markdown("Hemos generado una lista de libros especialmente para ti. Explora cada opción y marca tus favoritos.")
            st.caption("Haz clic en el corazón de los libros que más te llamen la atención.")

            favs = render_libros(st.session_state.recs["lista"],"unica")

            c1, c2 = st.columns([1,4])
            with c2:
                if st.button("Continuar", type="primary"):
                    st.session_state.du.update({"favoritos":" | ".join(favs)})
                    st.session_state.pantalla = "tam"; st.rerun()

        # ── TAM ────────────────────────────────────────────────────────────────
        elif pantalla == "tam":
            render_progress(3)
            st.markdown("## Cuestionario final")
            st.markdown("Queremos conocer tu opinión para seguir mejorando el sistema.")
            st.markdown("En una escala de 1 a 7, ¿qué tan de acuerdo estás con las siguientes afirmaciones?")

            etq_likert = ["Totalmente\nen desacuerdo","Moderadamente\nen desacuerdo","Algo en\ndesacuerdo","Neutral","Algo de\nacuerdo","Moderadamente\nde acuerdo","Totalmente\nde acuerdo"]

            def fila_tam(label, key):
                st.markdown(f"**{label}**")
                if key not in st.session_state:
                    st.session_state[key] = None
                cols = st.columns(7)
                for i, col in enumerate(cols):
                    n = i+1
                    with col:
                        tipo = "primary" if st.session_state[key] == n else "secondary"
                        if st.button(str(n), key=f"{key}_btn{n}", type=tipo, use_container_width=True):
                            st.session_state[key] = n
                            st.rerun()
                        st.markdown(f'<div style="font-size:9px;color:#94A3B8;text-align:center;line-height:1.2;">{etq_likert[i]}</div>', unsafe_allow_html=True)
                return st.session_state[key]

            pu1 = fila_tam("Usar este sistema mejora mi rendimiento académico.", "PU1")
            pu2 = fila_tam("Usar este sistema aumenta mi productividad.", "PU2")
            pu3 = fila_tam("Usar este sistema mejora mi efectividad en lo académico.", "PU3")
            pu4 = fila_tam("Considero que el sistema es útil para mis intereses académicos.", "PU4")

            peou1 = fila_tam("Mi interacción con el sistema es clara y entendible.", "PEOU1")
            peou2 = fila_tam("Interactuar con el sistema no requiere mucho de mi esfuerzo mental.", "PEOU2")
            peou3 = fila_tam("Encuentro que el sistema es fácil de usar.", "PEOU3")
            peou4 = fila_tam("Encuentro que es fácil hacer que el sistema haga lo que yo quiero que haga.", "PEOU4")

            rel1 = fila_tam("En mi vida académica, el uso del sistema es importante.", "REL1")
            rel2 = fila_tam("En mi vida académica, el uso del sistema es relevante.", "REL2")
            rel3 = fila_tam("El uso del sistema es pertinente para mis tareas relacionadas a mi vida académica.", "REL3")

            out1 = fila_tam("La calidad de las recomendaciones que obtengo del sistema es alta.", "OUT1")
            out2 = fila_tam("No tengo problema con la calidad de las recomendaciones del sistema.", "OUT2")
            out3 = fila_tam("Califico las recomendaciones del sistema como excelentes.", "OUT3")

            bi1 = fila_tam("Asumiendo que tuviera acceso al sistema, lo usaría.", "BI1")
            bi2 = fila_tam("Dado que tuviera acceso al sistema, predigo que lo usaría.", "BI2")
            meses = st.selectbox("Planeo usar el sistema en los próximos:", ["— Selecciona —","1 mes","3 meses","6 meses","12 meses","Más de 12 meses","No planeo usarlo"], key="BI3")

            st.session_state.du.update({"PU1":pu1,"PU2":pu2,"PU3":pu3,"PU4":pu4,"PEOU1":peou1,"PEOU2":peou2,"PEOU3":peou3,"PEOU4":peou4,"REL1":rel1,"REL2":rel2,"REL3":rel3,"OUT1":out1,"OUT2":out2,"OUT3":out3,"BI1":bi1,"BI2":bi2,"BI3":meses})

            c1, c2 = st.columns([3,1])
            with c1:
                if st.button("Volver", key="volver_tam"):
                    st.session_state.pantalla = "recomendaciones"; st.rerun()
            with c2:
                if st.button("Enviar respuestas", type="primary", use_container_width=True):
                    respuestas_likert = [pu1,pu2,pu3,pu4,peou1,peou2,peou3,peou4,rel1,rel2,rel3,out1,out2,out3,bi1,bi2]
                    if any(r is None for r in respuestas_likert) or meses == "— Selecciona —":
                        st.error("Por favor responde todas las preguntas del cuestionario antes de continuar.")
                    else:
                        with st.spinner("Guardando..."):
                            try:
                                sheet = conectar_sheets()
                                guardar(sheet, st.session_state.du)
                                st.session_state.pantalla = "gracias"; st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

        # ── GRACIAS ────────────────────────────────────────────────────────────
        elif pantalla == "gracias":
            render_progress(4)
            st.balloons()
            st.markdown("## ¡Gracias por tu participación!")
            st.markdown("Tus respuestas nos ayudan a construir un mejor sistema de recomendación bibliográfica para toda la comunidad Javeriana.")
            st.info("Este proyecto es desarrollado por estudiantes del programa de Ciencia de Datos de la Pontificia Universidad Javeriana — Bogotá.")
            if st.button("Volver al inicio"):
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()

if __name__ == "__main__":
    main()
