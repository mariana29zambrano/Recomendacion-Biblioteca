"""
Grid search de los pesos del hibrido del Modelo 1 (W_HIST/W_PERF/W_POP),
recalibrado sobre los datos reales de la tercera iteracion (directorio
institucional completo + historico extendido, ya filtrado a Bogota).

Por que recalibrar: los pesos optimos documentados en Modelado.ipynb
(seccion 10.1: W_HIST=0.70 en la config base del notebook, 0.40/0.20/0.40 en
la app desplegada) se calibraron sobre una matriz de interacciones mucho mas
dispersa -- la mayoria de participantes del estudio no tenian historial real
(score_historial casi siempre 0). Con el directorio completo, ~97% de los
usuarios tienen su "ID de estudiante" disponible y una fraccion mucho mayor
tiene historial real de prestamos. Burke (1999, ya citado en el documento de
grado) es explicito en que los pesos de un hibrido deben calibrarse
empiricamente -- si la densidad de la matriz cambia, el punto optimo se mueve.

Metodologia (replica exacta de Modelado.ipynb seccion 7-10.1):
  - construir_interacciones: peso = peso_base * (1 + peso_recencia), con
    peso_recencia = exp(-dias_desde_prestamo/180). En v3 el peso_base usa
    renewalCount graduado (1 + 0.1*renovaciones) en vez del binario 0.5x del
    notebook original -- consistente con la decision ya tomada en
    precompute_cache_v3.py, para que los pesos que se calibren aqui sean
    fieles a la formula que realmente corre en produccion.
  - Split leave-one-out temporal: la ultima interaccion (por fecha) de cada
    usuario se separa como test; el resto es train. Umbral minimo de 5
    obras distintas por usuario para ser evaluable (mismo umbral y misma
    justificacion que el notebook: con menos, no hay senal suficiente).
  - evaluar_modelo: MRR@10, NDCG@10, Coverage@10, Novelty@10, Diversity@10,
    exactamente las mismas metricas y formulas que Modelado.ipynb.

Proteccion de datos: mismo esquema que precompute_cache_v3.py -- usecols
restringe que columnas de identificacion personal real nunca se carguen a
memoria, y assert_sin_pii() es la red de seguridad adicional. Este script
solo imprime y grafica metricas agregadas de desempeno (NDCG, coverage,
etc.) sobre usuarios identificados por su UUID interno -- nunca nombres,
nunca texto de perfil de una persona en particular.

Corre localmente (puede tardar bastante por el tamano del catalogo -- se
imprime progreso por configuracion evaluada):
    python scripts/grid_search_pesos_v3.py
"""
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(__file__))
from precompute_cache import MAPEO_FAC, nt
from precompute_cache_v3 import (
    ACCIONES_VALIDAS_V3, CARPETA, DECODE_PATRON_GROUP, PATH_PRESTAMOS,
    PATH_USUARIOS, USUARIOS_COLS, PRESTAMOS_COLS, assert_sin_pii,
)

UMBRAL_MIN_INTERACCIONES = 5  # mismo umbral y justificacion que Modelado.ipynb
MUESTRA_TEST = int(os.environ.get("GS_MUESTRA_TEST", 500))  # mismo tamano que el notebook
TOP_N = 10
PASO_GRID = float(os.environ.get("GS_PASO_GRID", 0.1))  # resolucion (w_hist, w_pop)


def construir_interacciones_v3(df):
    """Misma logica de peso que precompute_cache_v3.inter_v3, aplicada aqui
    a un subconjunto train/test en vez de al historico completo."""
    df = df.copy()
    max_fecha = df["loanDate"].max()
    dias = (max_fecha - df["loanDate"]).dt.days.fillna(730)
    peso_recencia = np.exp(-dias / 180)
    peso_renovacion = 1 + 0.1 * df["renewalCount"].fillna(0)
    df["peso_evento"] = peso_renovacion * (1 + peso_recencia)

    inter = (
        df.groupby(["user_id", "instance_id"])
        .agg(peso=("peso_evento", "sum"), n=("action", "count"),
             facultad_clean=("facultad_clean", "first"),
             programa_clean=("programa_clean", "first"))
        .reset_index()
    )
    inter["peso"] = np.log1p(inter["peso"])
    return inter


def main():
    t0 = time.time()

    print("Cargando catalogo + TF-IDF ya precomputados (iteracion 2)...")
    catalogo_obras = pd.read_parquet("cache/obras.parquet")
    X_content = load_npz("cache/tfidf_matrix.npz")
    vectorizer = joblib.load("cache/tfidf_vectorizer.joblib")
    book_to_row = {iid: i for i, iid in enumerate(catalogo_obras["instance_id"])}
    n_items = len(catalogo_obras)
    # Normalizado una sola vez: cosine_similarity(a, X) normaliza X por dentro
    # en CADA llamada (recalcula la norma de las ~330k filas); como aqui se
    # llama miles de veces, se precalcula X_norm y se hace el producto punto
    # a mano (X_norm @ v_norm.T), matematicamente equivalente pero sin repetir
    # ese trabajo cada vez.
    X_norm = normalize(X_content)

    print("Leyendo usuarios/prestamos (solo columnas permitidas)...")
    usuarios = pd.read_csv(PATH_USUARIOS, usecols=USUARIOS_COLS, low_memory=False)
    assert_sin_pii(usuarios, "usuarios")
    prestamos = pd.read_csv(PATH_PRESTAMOS, usecols=PRESTAMOS_COLS, low_memory=False)
    assert_sin_pii(prestamos, "prestamos")

    # Mismo filtro de sede/accion que precompute_cache_v3.py
    es_cali = prestamos["patronGroupAtCheckout.name"].str.contains("Cali", case=False, na=False)
    es_sede_desconocida = prestamos["patronGroupAtCheckout.name"].isna()
    prestamos = prestamos[~es_cali & ~es_sede_desconocida].copy()
    prestamos = prestamos[prestamos["action"].isin(ACCIONES_VALIDAS_V3)].copy()
    prestamos["loanDate"] = pd.to_datetime(prestamos["loanDate"], errors="coerce", utc=True)
    prestamos = prestamos.dropna(subset=["loanDate", "userId", "item.instanceId"]).copy()

    facultad_rellena = usuarios["customFields.facultad"].fillna("")
    usuarios["facultad_clean"] = facultad_rellena.map(MAPEO_FAC).fillna(facultad_rellena.apply(nt))
    usuarios["programa_clean"] = usuarios["customFields.programa_2"].fillna("").apply(nt)
    usuarios["tipo_usuario"] = usuarios["patronGroup"].map(DECODE_PATRON_GROUP).fillna("")

    eval_base = prestamos.rename(columns={"userId": "user_id", "item.instanceId": "instance_id"})
    eval_base = eval_base[eval_base["instance_id"].isin(book_to_row.keys())]
    eval_base = eval_base.merge(
        usuarios[["id", "facultad_clean", "programa_clean", "tipo_usuario"]],
        left_on="user_id", right_on="id", how="left"
    )
    eval_base = eval_base.sort_values(["user_id", "loanDate"])

    usuarios_validos = eval_base.groupby("user_id")["instance_id"].nunique()
    usuarios_validos = usuarios_validos[usuarios_validos >= UMBRAL_MIN_INTERACCIONES].index
    eval_base = eval_base[eval_base["user_id"].isin(usuarios_validos)]
    print(f"Usuarios evaluables (>= {UMBRAL_MIN_INTERACCIONES} obras distintas en Bogota): {len(usuarios_validos):,}")

    test_idx = eval_base.groupby("user_id").tail(1).index
    test = eval_base.loc[test_idx].copy()
    train = eval_base.drop(test_idx).copy()

    vistos_train = train.groupby("user_id")["instance_id"].apply(set).to_dict()
    mask_valido = test.apply(
        lambda r: r["instance_id"] not in vistos_train.get(r["user_id"], set()), axis=1
    )
    test = test[mask_valido].copy()
    train = train[train["user_id"].isin(test["user_id"])].copy()
    print(f"Train: {len(train):,} interacciones  |  Test: {len(test):,} usuarios evaluables")

    interacciones_train = construir_interacciones_v3(train)

    usuarios_train = usuarios[usuarios["id"].isin(train["user_id"])].set_index("id")
    perfil_dict_train = (
        (usuarios_train["facultad_clean"] + " " + usuarios_train["programa_clean"] + " " + usuarios_train["tipo_usuario"])
        .str.strip()
        .to_dict()
    )

    # Vectores densos de popularidad por facultad/programa (solo para las
    # claves que de verdad aparecen en train -- no para las ~1600 posibles).
    # Precomputados una sola vez; antes se re-filtraba/mergeaba una tabla
    # contra las ~330k obras del catalogo en CADA llamada de recomendacion.
    def construir_vectores_pop(df_pop, col_key):
        vec = {}
        for key, grp in df_pop.groupby(col_key):
            arr = np.zeros(n_items, dtype=np.float32)
            idxs = grp["instance_id"].map(book_to_row)
            validos = idxs.notna()
            arr[idxs[validos].astype(int).values] = grp.loc[validos, "score_pop"].values
            vec[key] = arr
        return vec

    pop_fac_train = (
        interacciones_train.groupby(["facultad_clean", "instance_id"])
        .agg(n_usuarios=("user_id", "nunique")).reset_index()
    )
    pop_fac_train["score_pop"] = np.log1p(pop_fac_train["n_usuarios"])
    pop_prog_train = (
        interacciones_train.groupby(["programa_clean", "instance_id"])
        .agg(n_usuarios=("user_id", "nunique")).reset_index()
    )
    pop_prog_train["score_pop"] = np.log1p(pop_prog_train["n_usuarios"])
    pop_fac_vec = construir_vectores_pop(pop_fac_train, "facultad_clean")
    pop_prog_vec = construir_vectores_pop(pop_prog_train, "programa_clean")

    facultad_por_id = usuarios.set_index("id")["facultad_clean"].to_dict()
    programa_por_id = usuarios.set_index("id")["programa_clean"].to_dict()

    def construir_vectores_usuario(user_ids):
        """Precomputa, UNA SOLA VEZ por usuario (no por configuracion), los
        tres vectores de score que no dependen de los pesos del hibrido:
        score_historial, score_perfil, score_pop_norm. Evaluar 66+
        configuraciones recalculando esto en cada una (como en el primer
        intento de este script) era el cuello de botella real: cada llamada
        implica 2 productos contra las ~330k obras del catalogo."""
        # Agrupado una sola vez (en vez de filtrar el DataFrame completo por
        # cada usuario): interacciones_train ya viene de train, que a su vez
        # viene de eval_base, ya filtrado a instance_id en book_to_row -- no
        # hace falta repetir ese isin() por usuario.
        grupos_por_usuario = {uid: g for uid, g in interacciones_train.groupby("user_id")}

        cache = {}
        for user_id in set(user_ids):
            historial = grupos_por_usuario.get(user_id, interacciones_train.iloc[0:0])
            score_historial = np.zeros(n_items, dtype=np.float32)
            vistos_idx = np.array([], dtype=int)
            if not historial.empty:
                indices = historial["instance_id"].map(book_to_row).values.astype(int)
                vistos_idx = indices
                pesos = historial["peso"].values
                uv = X_content[indices].multiply(pesos.reshape(-1, 1)).sum(axis=0)
                uv = csr_matrix(uv / pesos.sum())
                uv_norm = normalize(uv)
                score_historial = np.asarray(X_norm.dot(uv_norm.T).todense()).ravel()

            profile_text = perfil_dict_train.get(user_id, "")
            score_perfil = np.zeros(n_items, dtype=np.float32)
            if profile_text.strip():
                pv = vectorizer.transform([profile_text])
                pv_norm = normalize(pv)
                score_perfil = np.asarray(X_norm.dot(pv_norm.T).todense()).ravel()

            facultad_usuario = facultad_por_id.get(user_id, "")
            programa_usuario = programa_por_id.get(user_id, "")
            score_pop_fac = pop_fac_vec.get(facultad_usuario, np.zeros(n_items, dtype=np.float32))
            score_pop_prog = pop_prog_vec.get(programa_usuario, np.zeros(n_items, dtype=np.float32)) if programa_usuario else np.zeros(n_items, dtype=np.float32)
            usa_programa = score_pop_prog.sum() > 0
            score_pop_base = score_pop_prog if usa_programa else score_pop_fac
            max_pop = score_pop_base.max()
            score_pop_norm = (score_pop_base / max_pop) if max_pop > 0 else np.zeros(n_items, dtype=np.float32)

            cache[user_id] = (score_historial, score_perfil, score_pop_norm, vistos_idx)
        return cache

    # Invariantes a lo largo de todo el grid search (no dependen de los
    # pesos) -- se calculan una sola vez en vez de por cada usuario de test
    # en cada configuracion, que es como estaba en Modelado.ipynb original.
    popularidad_global = interacciones_train.groupby("instance_id")["user_id"].nunique()
    n_users_total = interacciones_train["user_id"].nunique()
    popularidad_global_arr = np.ones(n_items)
    for iid, idx in book_to_row.items():
        popularidad_global_arr[idx] = popularidad_global.get(iid, 1)

    def evaluar_modelo_v3(w_hist, w_perf, w_pop, test_df, vectores_cache):
        todos_recomendados = []
        detalle = []
        for _, row in test_df.iterrows():
            user_id, item_real = row["user_id"], row["instance_id"]
            score_historial, score_perfil, score_pop_norm, vistos_idx = vectores_cache[user_id]

            score_total = w_hist * score_historial + w_perf * score_perfil + w_pop * score_pop_norm
            if len(vistos_idx):
                score_total = score_total.copy()
                score_total[vistos_idx] = -np.inf

            top_idx = np.argpartition(-score_total, TOP_N)[:TOP_N]
            top_idx = top_idx[np.argsort(-score_total[top_idx])]
            recomendados = catalogo_obras["instance_id"].values[top_idx].tolist()
            todos_recomendados.extend(recomendados)

            hit = 1 if item_real in recomendados else 0
            rank = recomendados.index(item_real) + 1 if hit else None
            reciprocal_rank = 1 / rank if hit else 0
            ndcg_val = 1 / np.log2(rank + 1) if hit else 0

            novelty_val = np.mean([
                -np.log2(popularidad_global_arr[book_to_row[it]] / n_users_total + 1e-10) for it in recomendados
            ]) if recomendados else 0

            diversity_val = 0
            if len(recomendados) > 1:
                sim_mat = cosine_similarity(X_content[top_idx])
                n = len(top_idx)
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                diversity_val = 1 - sim_mat[mask].mean()

            detalle.append({"hit": hit, "reciprocal_rank": reciprocal_rank,
                             "ndcg": ndcg_val, "novelty": novelty_val, "diversity": diversity_val})

        detalle = pd.DataFrame(detalle)
        coverage = len(set(todos_recomendados)) / catalogo_obras["instance_id"].nunique()
        return {
            "MRR@10": detalle["reciprocal_rank"].mean(),
            "NDCG@10": detalle["ndcg"].mean(),
            "Coverage@10": coverage,
            "Novelty@10": detalle["novelty"].mean(),
            "Diversity@10": detalle["diversity"].mean(),
        }

    test_muestra = test.sample(n=min(MUESTRA_TEST, len(test)), random_state=42)
    print(f"Evaluando sobre muestra de test: {len(test_muestra):,} usuarios")

    print("Precomputando vectores de score por usuario (una sola vez, no por configuracion)...")
    tpre0 = time.time()
    vectores_cache = construir_vectores_usuario(test_muestra["user_id"].values)
    print(f"  listo en {time.time()-tpre0:.0f}s")

    pasos = np.round(np.arange(0, 1.0001, PASO_GRID), 2)
    resultados = []
    combos = [(wh, wpo) for wh in pasos for wpo in pasos if 0 <= round(1 - wh - wpo, 2) <= 1]
    print(f"Configuraciones a evaluar: {len(combos)}")

    for i, (w_hist, w_pop) in enumerate(combos, 1):
        w_perf = round(1 - w_hist - w_pop, 2)
        tc0 = time.time()
        met = evaluar_modelo_v3(w_hist, w_perf, w_pop, test_muestra, vectores_cache)
        met.update({"w_hist": w_hist, "w_perf": w_perf, "w_pop": w_pop})
        resultados.append(met)
        print(f"  [{i}/{len(combos)}] w_hist={w_hist} w_perf={w_perf} w_pop={w_pop} "
              f"-> NDCG@10={met['NDCG@10']:.4f} ({time.time()-tc0:.0f}s)")

    df_pesos = pd.DataFrame(resultados)
    df_pesos.to_csv("gs_pesos_m1_v3.csv", index=False)

    mejor = df_pesos.loc[df_pesos["NDCG@10"].idxmax()]
    print(f"\nMejor configuracion por NDCG@10: w_hist={mejor['w_hist']} "
          f"w_perf={mejor['w_perf']} w_pop={mejor['w_pop']} -> NDCG@10={mejor['NDCG@10']:.4f}")

    pivot = df_pesos.pivot(index="w_pop", columns="w_hist", values="NDCG@10")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Grid search de pesos — Modelo 1 (tercera iteración)\nNDCG@10 por (w_hist, w_pop), w_perf = 1 - w_hist - w_pop")
    plt.xlabel("w_hist"); plt.ylabel("w_pop")
    plt.tight_layout()
    plt.savefig("gs_pesos_m1_v3_heatmap.png", dpi=150, bbox_inches="tight")
    print("Guardado: gs_pesos_m1_v3_heatmap.png, gs_pesos_m1_v3.csv")
    print(f"Listo en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
