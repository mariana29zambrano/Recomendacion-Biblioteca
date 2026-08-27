import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import base64
from datetime import datetime
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
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

/* Tarjeta contenedora */
.card {
    background: white; border-radius: 16px;
    padding: 32px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Botones */
.btn-primary {
    background: #002147; color: white; border: none;
    padding: 12px 32px; border-radius: 8px; font-size: 14px;
    font-weight: 600; cursor: pointer; float: right;
}
.btn-secondary {
    background: white; color: #374151; border: 1.5px solid #E2E8F0;
    padding: 12px 24px; border-radius: 8px; font-size: 14px;
    font-weight: 500; cursor: pointer;
}

/* Libros */
.book-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid #F1F5F9;
}
.book-row:last-child { border-bottom: none; }
.book-info { flex: 1; }
.book-title { font-size: 13px; font-weight: 600; color: #1E293B; margin-bottom: 2px; }
.book-author { font-size: 12px; color: #64748B; }
.book-heart { font-size: 20px; cursor: pointer; color: #CBD5E1; padding: 4px 8px; }
.book-heart.liked { color: #EF4444; }

/* Lista header */
.lista-col-header {
    font-size: 13px; font-weight: 700; color: #002147;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 2px solid #002147;
}

/* Tabla de cuestionario */
.tam-table { width: 100%; border-collapse: collapse; margin-top: 8px; }
.tam-table th {
    text-align: center; font-size: 11px; font-weight: 600;
    color: #64748B; padding: 6px 4px;
    background: #F8FAFC; border-bottom: 1px solid #E2E8F0;
}
.tam-table td {
    padding: 12px 4px; border-bottom: 1px solid #F1F5F9;
    font-size: 13px; vertical-align: middle;
}
.tam-table td:first-child {
    color: #374151; font-weight: 500; padding-right: 16px; width: 55%;
}
.tam-table td { text-align: center; }
.tam-section-title {
    font-size: 14px; font-weight: 700; color: #002147;
    margin: 24px 0 4px 0; border-left: 3px solid #002147;
    padding-left: 10px;
}
.tam-section-caption { font-size: 11px; color: #94A3B8; margin-bottom: 8px; }

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

COL_INSTANCE_ID = "Instances - Instance UUID"
COL_TITLE       = "Instances - Index title"
COL_CONTRIBUTORS= "Instances - Contributors"
COL_SUBJECTS    = "Instances - Subject headings"
COL_PUBLICATION = "Instances - Publication"
COL_CALL_NUMBER = "Items - Item call number"
COL_MTYPE       = "Material type - Name"
COL_LOCATION    = "Effective location - Name"
COL_BARCODE     = "Items - Barcode"

# Pesos del score híbrido del Modelo 1 (óptimo empírico de la sección 10.1 del
# notebook, gs_pesos_m1_heatmap.png: mejor Recall@10 y NDCG@10 simultáneamente
# con w_pop=0.4 fijo y w_hist=0.4/w_perf=0.2 sobre el remanente).
W_HIST = 0.40
W_PERF = 0.20
W_POP  = 0.40
LOAN_USER    = "ID Usuario"
LOAN_BARCODE = "Item barcode"
LOAN_DATE    = "Date"
LOAN_ACTION  = "Circ action"
LOAN_PROFILE = "Perfil"
LOAN_PROGRAM = "Programa"
LOAN_FACULTY = "Facultad"
PERFIL_ESTUDIANTE = "Bogota Estudiantes"

ACCIONES_VALIDAS = ["Checked out","Checked out through override","Renewed","Renewed through override"]

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
        sheet.update([["timestamp","carnet","edad","perfil","facultad","programa","semestre_cargo","genero","tiene_prestamos","titulo_recordado","algoritmo","favoritos","PU1","PU2","PU3","PU4","PEOU1","PEOU2","PEOU3","PEOU4","REL1","REL2","REL3","OUT1","OUT2","OUT3","BI1","BI2","BI3"]], "A1")
    return sheet

def guardar(sheet, d):
    sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        d.get("carnet",""),d.get("edad",""),d.get("perfil",""),d.get("facultad",""),d.get("programa",""),
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

# IDs de Google Drive de los CSV pesados (no versionados en el repo, ver CLAUDE.md).
ID_COLECCION = "1_qYRsiK9njZnciuQERVaLo22dGnt32s6"
ID_PRESTAMOS = "1-TXpnJlUsGPT7spJ4ZFUQcsxRa21U68n"

# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando la colección bibliográfica...")
def cargar_datos():
    import warnings, os, gdown
    warnings.filterwarnings("ignore")

    path_col = "data/Biblioteca_General.csv"
    path_pre = "data/Prestamos_completo.csv"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(path_col):
        gdown.download(f"https://drive.google.com/uc?id={ID_COLECCION}", path_col, quiet=False)
    if not os.path.exists(path_pre):
        gdown.download(f"https://drive.google.com/uc?id={ID_PRESTAMOS}", path_pre, quiet=False)

    col=pd.read_csv(path_col,low_memory=False)
    pre=pd.read_csv(path_pre,low_memory=False)

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

    col[COL_BARCODE]=col[COL_BARCODE].astype(str).str.strip()
    col[COL_INSTANCE_ID]=col[COL_INSTANCE_ID].astype(str)
    for c in [COL_TITLE,COL_SUBJECTS,COL_PUBLICATION,COL_CALL_NUMBER,COL_MTYPE,COL_LOCATION]:
        col[c]=col[c].fillna("").astype(str)
    col[COL_CONTRIBUTORS]=col[COL_CONTRIBUTORS].fillna("").astype(str)
    col["contributors_text"]=col[COL_CONTRIBUTORS].apply(contrib)
    col["publication_text"]=col[COL_PUBLICATION].apply(publication_text)

    obras=(col.groupby(COL_INSTANCE_ID)
           .agg({COL_TITLE:"first",COL_SUBJECTS:ju,"publication_text":ju,"contributors_text":ju,
                 COL_CALL_NUMBER:"first",COL_MTYPE:"first",COL_LOCATION:"first"})
           .reset_index().rename(columns={COL_INSTANCE_ID:"instance_id"}))
    b2i=(col[[COL_BARCODE,COL_INSTANCE_ID]].dropna().drop_duplicates()
         .set_index(COL_BARCODE)[COL_INSTANCE_ID].to_dict())

    sw_es=["de","la","el","los","las","y","en","del","a","por","para","con","una","un","al","se","su","sus","como","mas","o","e","que","es","sobre","entre","sin","edicion","vol","ed"]
    sw_pt=["de","da","do","das","dos","e","em","um","uma","para","com","por","que","se","na","no","nas","nos","ao","aos"]
    sw_fr=["de","la","le","les","et","en","du","des","un","une","par","sur","dans","avec","pour","au","aux"]
    sw=list(set(sw_es+sw_pt+sw_fr+list(ENGLISH_STOP_WORDS)))
    obras["subjects_clean"]=obras[COL_SUBJECTS].str.replace(";", " ", regex=False)
    obras["content_text"]=(obras[COL_TITLE]+" "+obras["subjects_clean"]+" "+obras["subjects_clean"]+" "+
                            obras["contributors_text"]+" "+obras["publication_text"]+" "+
                            obras[COL_MTYPE]+" "+obras[COL_LOCATION]).apply(nt)
    vec=TfidfVectorizer(max_features=50000,ngram_range=(1,2),min_df=2,max_df=0.85,stop_words=sw)
    X=vec.fit_transform(obras["content_text"])
    book_to_row=pd.Series(obras.index.values,index=obras["instance_id"]).to_dict()

    pre[LOAN_USER]=pre[LOAN_USER].astype(str).str.strip()
    pre[LOAN_BARCODE]=pre[LOAN_BARCODE].astype(str).str.strip()
    pre[LOAN_DATE]=pd.to_datetime(pre[LOAN_DATE],errors="coerce")
    pre=pre.dropna(subset=[LOAN_DATE])
    pre=pre[pre[LOAN_ACTION].isin(ACCIONES_VALIDAS)].copy()
    pre["instance_id"]=pre[LOAN_BARCODE].map(b2i)
    pre=pre.dropna(subset=["instance_id"]).copy()
    mf=pre[LOAN_DATE].max()
    pre["dias"]=(mf-pre[LOAN_DATE]).dt.days.fillna(730)
    pre["peso_recencia"]=np.exp(-pre["dias"]/180)
    pre["peso_base"]=np.where(pre[LOAN_ACTION].str.contains("Renewed",case=False),0.5,1.0)
    pre["peso_evento"]=pre["peso_base"]*(1+pre["peso_recencia"])
    pre[LOAN_FACULTY]=pre[LOAN_FACULTY].fillna("").astype(str)
    pre[LOAN_PROFILE]=pre[LOAN_PROFILE].fillna("").astype(str)
    pre[LOAN_PROGRAM]=pre[LOAN_PROGRAM].fillna("").astype(str)

    inter=(pre.rename(columns={LOAN_USER:"user_id",LOAN_BARCODE:"barcode",LOAN_FACULTY:"facultad",
                                LOAN_PROFILE:"perfil_prestamo",LOAN_PROGRAM:"programa_prestamo"})
           .groupby(["user_id","instance_id"])
           .agg(peso=("peso_evento","sum"),n=("barcode","count"),facultad=("facultad","first"),
                perfil_prestamo=("perfil_prestamo","first"),programa_prestamo=("programa_prestamo","first"))
           .reset_index())
    inter["peso"]=np.log1p(inter["peso"])
    inter["facultad_clean"]=inter["facultad"].map(MAPEO_FAC).fillna("")
    inter["programa_clean"]=inter["programa_prestamo"].apply(nt)

    pop_fac=(inter.groupby(["facultad_clean","instance_id"])
             .agg(n_usuarios=("user_id","nunique")).reset_index())
    pop_fac["score_pop"]=np.log1p(pop_fac["n_usuarios"])
    pop_gen=(inter.groupby("instance_id")["user_id"].nunique()
             .reset_index(name="n").sort_values("n",ascending=False))

    # Popularidad por programa (carrera), solo estudiantes, para no mezclar prestamos de
    # profesores/administrativos/staff en la senal de "que leen mis companeros de programa".
    # Fallback a pop_fac se resuelve en rec_A cuando el programa no tiene senal propia.
    pop_prog=(inter[inter["perfil_prestamo"]==PERFIL_ESTUDIANTE]
              .groupby(["programa_clean","instance_id"])
              .agg(n_usuarios=("user_id","nunique")).reset_index())
    pop_prog["score_pop"]=np.log1p(pop_prog["n_usuarios"])

    return {"obras":obras,"inter":inter,"pop_fac":pop_fac,"pop_prog":pop_prog,"pop_gen":pop_gen,
            "vec":vec,"X":X,"book_to_row":book_to_row}

# ── Recomendaciones ────────────────────────────────────────────────────────────
def rec_A(fac,prog,datos,n=TOP_N,carnet=""):
    """Modelo 1 (TF-IDF híbrido) — score = W_HIST*historial + W_PERF*perfil + W_POP*popularidad,
    pesos óptimos de la sección 10.1 del notebook (gs_pesos_m1_heatmap.png).
    score_perfil usa facultad+programa como proxy del perfil académico: el notebook lo construye
    desde un archivo "BD Digitales" que no forma parte de este proyecto. score_historial solo
    aporta si el carnet ingresado coincide con un "ID Usuario" real en Prestamos_completo.csv
    (poco probable para un participante nuevo/anónimo del estudio); si no hay coincidencia,
    su contribución es naturalmente cero y el score se apoya en perfil + popularidad.
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

    txt=re.sub(r"[^a-z0-9\s]"," ",unidecode(f"{fac} {prog}".lower())).strip()
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
    return cand.sort_values("score",ascending=False).head(n)[["instance_id",COL_TITLE,COL_SUBJECTS,COL_MTYPE,COL_CALL_NUMBER,"contributors_text"]].reset_index(drop=True)

def rec_C(titulo,datos,n=TOP_N):
    obras=datos["obras"]; vec=datos["vec"]; X=datos["X"]
    if not titulo or not titulo.strip():
        pg=datos["pop_gen"].head(n)
        return obras[obras["instance_id"].isin(pg["instance_id"])][["instance_id",COL_TITLE,COL_SUBJECTS,COL_MTYPE,COL_CALL_NUMBER,"contributors_text"]].reset_index(drop=True)
    txt=re.sub(r"[^a-z0-9\s]"," ",unidecode(titulo.lower())).strip()
    sims=cosine_similarity(vec.transform([txt]),X).ravel()
    cand=obras.copy(); cand["score"]=sims
    return cand.sort_values("score",ascending=False).iloc[1:n+1][["instance_id",COL_TITLE,COL_SUBJECTS,COL_MTYPE,COL_CALL_NUMBER,"contributors_text"]].reset_index(drop=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
def _logo_puj_b64():
    with open("Logo PUJ.png", "rb") as f:
        return base64.b64encode(f.read()).decode()

def render_sidebar(pantalla):
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

# ── Tabla TAM ──────────────────────────────────────────────────────────────────
def tam_radio(label, key, opciones):
    """Radio horizontal sin label visible arriba."""
    return st.radio(label, opciones, horizontal=True, key=key,
                    label_visibility="visible")

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    for k,v in [("pantalla","consentimiento"),("du",{}),("recs",{})]:
        if k not in st.session_state: st.session_state[k]=v

    pantalla = st.session_state.pantalla
    render_sidebar(pantalla)

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
- El número de carnet es opcional y solo para control interno
- Los datos podrán utilizarse en futuras iteraciones del proyecto

Los datos de este formulario serán usados exclusivamente para la inscripción al evento. Puede consultar la Política de Protección de Datos Personales de la Universidad en la página web www.javeriana.edu.co. El canal de comunicación para revocar la autorización otorgada o solicitar la supresión de los datos es el correo electrónico: usodedatos@javeriana.edu.co

*Al continuar autorizas el uso de la información ingresada bajo las condiciones descritas.*
                """)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Acepto y deseo participar", type="primary", use_container_width=True):
                        st.session_state.pantalla = "perfil"; st.rerun()
                with c2:
                    if st.button("No deseo participar", use_container_width=True):
                        st.info("Gracias. Puedes cerrar esta ventana.")

        # ── PERFIL ─────────────────────────────────────────────────────────────
        elif pantalla == "perfil":
            render_progress(1)
            st.markdown("## Cuéntanos sobre ti")
            st.caption("Esta información nos permite personalizar tus recomendaciones y mejorar el sistema en el futuro.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Datos de identificación**")
                carnet = st.text_input("Número de carnet (solo para control interno):", placeholder="Ej: 00123456")
                edad   = st.text_input("Edad:", placeholder="Ej: 22")
                genero = st.selectbox("Género:", ["Prefiero no decirlo","Femenino","Masculino","No binario","Otro"])
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
                        st.session_state.du = {"carnet":carnet,"edad":edad,"perfil":perfil,"facultad":facultad,"programa":programa,"semestre_cargo":semestre_cargo,"genero":genero,"tiene_prestamos":tiene,"titulo_recordado":titulo_rec}
                        with st.spinner("Generando tus recomendaciones..."):
                            datos = cargar_datos()
                            du    = st.session_state.du
                            if du["tiene_prestamos"] == "Sí":
                                algoritmo = "C"
                                lista = rec_C(du.get("titulo_recordado",""),datos)
                            else:
                                algoritmo = "A"
                                lista = rec_A(du["facultad"],du["programa"],datos,carnet=du.get("carnet",""))
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
                    st.session_state[key] = 1
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

            pu1 = fila_tam("Las recomendaciones fueron relevantes para mis intereses académicos.", "PU1")
            pu2 = fila_tam("La información de los libros me ayudó a tomar una buena decisión.", "PU2")
            pu3 = fila_tam("El sistema fue útil en mi vida académica.", "PU3")
            pu4 = fila_tam("Usar este sistema aumentaría mi productividad académica.", "PU4")

            peou1 = fila_tam("Mi interacción con el sistema fue clara y comprensible.", "PEOU1")
            peou2 = fila_tam("Interactuar con el sistema no requirió mucho esfuerzo mental.", "PEOU2")
            peou3 = fila_tam("El sistema me resultó fácil de usar.", "PEOU3")
            peou4 = fila_tam("Me resultó fácil lograr que el sistema hiciera lo que quería.", "PEOU4")

            rel1 = fila_tam("En mis actividades académicas, el uso del sistema es importante.", "REL1")
            rel2 = fila_tam("En mis actividades académicas, el uso del sistema es relevante.", "REL2")
            rel3 = fila_tam("El uso del sistema es pertinente para mis diversas actividades académicas.", "REL3")

            out1 = fila_tam("La calidad de las recomendaciones que obtengo del sistema es alta.", "OUT1")
            out2 = fila_tam("No tengo problema con la calidad de las recomendaciones del sistema.", "OUT2")
            out3 = fila_tam("Considero que los resultados del sistema son excelentes.", "OUT3")

            bi1 = fila_tam("Asumiendo que tuviera acceso al sistema, tengo intención de usarlo.", "BI1")
            bi2 = fila_tam("Dado que tuviera acceso al sistema, predigo que lo usaría.", "BI2")
            meses = st.selectbox("Planeo usar el sistema en los próximos:", ["1 mes","3 meses","6 meses","12 meses","Más de 12 meses","No planeo usarlo"], key="BI3")

            st.session_state.du.update({"PU1":pu1,"PU2":pu2,"PU3":pu3,"PU4":pu4,"PEOU1":peou1,"PEOU2":peou2,"PEOU3":peou3,"PEOU4":peou4,"REL1":rel1,"REL2":rel2,"REL3":rel3,"OUT1":out1,"OUT2":out2,"OUT3":out3,"BI1":bi1,"BI2":bi2,"BI3":meses})

            c1, c2 = st.columns([3,1])
            with c1:
                if st.button("Volver", key="volver_tam"):
                    st.session_state.pantalla = "recomendaciones"; st.rerun()
            with c2:
                if st.button("Enviar respuestas", type="primary", use_container_width=True):
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
