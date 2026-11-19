"""Лабораторная работа 5. Кластеризация категориальных данных. Алгоритм CLOPE.

Единый файл с решением в порядке пунктов задания:
1. Демонстрационный пример Loginom «Кластеризация транзакций» (набор Mushroom).
2. Описание набора Heart Disease (UCI, Cleveland).
3. Дескриптивный анализ.
4. Преобразование данных: дискретизация и перевод в транзакции.
5. CLOPE: подбор r, интерпретация, метрики, значимость признаков.
6. Мера Гауэра и PAM (k-medoids), контрольный k-prototypes.
7. Сравнение решений и сохранение результатов.

Функции определены перед пунктом, в котором впервые используются; main выполняет
все пункты по порядку. Запуск: python lr5_clope_heart.py (рисунки — figures/, таблицы — results/).
"""
import itertools
import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from scipy import stats
from sklearn import metrics
from sklearn.manifold import MDS

IN_NOTEBOOK = "ipykernel" in sys.modules
if IN_NOTEBOOK:
    from IPython.display import display
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(globals().get("__file__", "lr5_clope_heart.py")).resolve().parent
DATA, FIG, RES = ROOT / "data", ROOT / "figures", ROOT / "results"
SUMMARY = {}
warnings.filterwarnings("ignore")
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "axes.titlesize": 13,
                     "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})
sns.set_palette("colorblind")


def savefig(name):
    """Сохраняет текущий рисунок в figures/<name>.png; в ноутбуке показывает его, в скрипте закрывает."""
    plt.savefig(FIG / f"{name}.png")
    plt.show() if IN_NOTEBOOK else plt.close()


def show(obj, digits=None):
    """Выводит таблицу: в ноутбуке через display, в скрипте через print."""
    obj = obj if digits is None else obj.round(digits)
    display(obj) if IN_NOTEBOOK else print(obj.to_string(), end="\n\n")


class Cluster:
    """Кластер CLOPE: число транзакций N, площадь S, гистограмма элементов Occ и ширина W."""

    def __init__(self):
        """Создаёт пустой кластер (N = 0, S = 0, Occ = {})."""
        self.n, self.s, self.occ = 0, 0, {}

    @property
    def w(self):
        """Ширина кластера W — число различных элементов."""
        return len(self.occ)

    def add(self, t):
        """Добавляет транзакцию t, обновляя N, S и гистограмму Occ."""
        self.n += 1
        self.s += len(t)
        for i in t:
            self.occ[i] = self.occ.get(i, 0) + 1

    def remove(self, t):
        """Удаляет транзакцию t из кластера, обновляя N, S и Occ."""
        self.n -= 1
        self.s -= len(t)
        for i in t:
            self.occ[i] -= 1
            if not self.occ[i]:
                del self.occ[i]


def delta_add(cl, t, r):
    """Приращение числителя Profit при добавлении транзакции t в кластер cl.

    DeltaAdd = S_new * (N + 1) / W_new ** r - S * N / W ** r, где S_new = S + |t|,
    W_new = W + число элементов t, отсутствующих в cl; для пустого кластера |t| / |t| ** r.
    """
    s_new = cl.s + len(t)
    w_new = cl.w + sum(1 for i in t if i not in cl.occ)
    if cl.n == 0:
        return s_new / (w_new ** r)
    return s_new * (cl.n + 1) / (w_new ** r) - cl.s * cl.n / (cl.w ** r)


def _best_cluster(clusters, t, r, max_clusters):
    """Индекс кластера с максимальным DeltaAdd или None, если выгоднее открыть новый кластер."""
    best_i, best_d = None, float("-inf")
    for idx, cl in enumerate(clusters):
        if cl.n > 0 and (d := delta_add(cl, t, r)) > best_d:
            best_i, best_d = idx, d
    n_active = sum(1 for c in clusters if c.n > 0)
    if (max_clusters is None or n_active < max_clusters) and len(t) / (len(set(t)) ** r) > best_d:
        return None
    return best_i


def profit(clusters, r):
    """Глобальный критерий Profit(C, r) = sum(S_i * N_i / W_i ** r) / sum(N_i)."""
    active = [c for c in clusters if c.n > 0]
    return sum(c.s * c.n / (c.w ** r) for c in active) / sum(c.n for c in active)


def clope(transactions, r=2.6, max_iter=50, max_clusters=None, seed=None):
    """Кластеризация транзакций алгоритмом CLOPE (Yang, Guan, You, 2002).

    Фаза 1 (инициализация): каждая транзакция по очереди помещается в кластер
    с максимальным DeltaAdd (или в новый кластер). Фаза 2 (итерации): каждая
    транзакция извлекается из своего кластера и снова размещается по максимуму
    DeltaAdd; проходы повторяются, пока происходят перемещения (не более max_iter).
    max_clusters — ограничение числа кластеров (как в Loginom); seed — случайный
    порядок просмотра транзакций (None — исходный порядок).

    Возвращает (labels, profit, history, n_iter): номера кластеров (0 — самый
    крупный), итоговый Profit, Profit после инициализации и каждой итерации,
    число итераций фазы 2.
    """
    order = list(range(len(transactions)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    clusters, labels = [], [None] * len(transactions)

    def place(j):
        """Размещает транзакцию j в лучший кластер; возвращает True, если она сменила кластер."""
        t, cur = transactions[j], labels[j]
        if cur is not None:
            clusters[cur].remove(t)
        b = _best_cluster(clusters, t, r, max_clusters)
        if b is None:
            if cur is not None and clusters[cur].n == 0:
                b = cur
            else:
                clusters.append(Cluster())
                b = len(clusters) - 1
        clusters[b].add(t)
        labels[j] = b
        return b != cur

    for j in order:
        place(j)
    history = [profit(clusters, r)]
    for n_iter in range(1, max_iter + 1):
        moved = [place(j) for j in order]
        history.append(profit(clusters, r))
        if not any(moved):
            break
    remap = {old: new for new, (old, _) in enumerate(Counter(labels).most_common())}
    return [remap[lab] for lab in labels], history[-1], history, n_iter


def build_clusters(transactions, labels):
    """Восстанавливает объекты Cluster по транзакциям и меткам кластеров."""
    clusters = {}
    for t, lab in zip(transactions, labels):
        clusters.setdefault(lab, Cluster()).add(t)
    return clusters


def cluster_stats(transactions, labels):
    """Параметры кластеров, как на выходе «Параметры кластеров» Loginom: N, W, S, H = S / W, G = S / W ** 2."""
    return [{"cluster": k, "N": c.n, "W": c.w, "S": c.s, "H": c.s / c.w, "G": c.s / c.w ** 2}
            for k, c in sorted(build_clusters(transactions, labels).items())]


def external_metrics(y_true, labels):
    """Внешние метрики относительно эталона: ARI, NMI, однородность, полнота, V-мера и чистота (Purity)."""
    h, c, v = metrics.homogeneity_completeness_v_measure(y_true, labels)
    ct = pd.crosstab(np.asarray(labels), np.asarray(y_true))
    return {"ARI": metrics.adjusted_rand_score(y_true, labels),
            "NMI": metrics.normalized_mutual_info_score(y_true, labels),
            "Homogeneity": h, "Completeness": c, "V-measure": v,
            "Purity": ct.max(axis=1).sum() / ct.values.sum()}


def mushroom_transactions(df, with_class=True):
    """Транзакции демо-примера Loginom из элементов «признак=значение» («?» — обычная категория).

    Аналог узлов «Добавление ID» → «Свёртка столбцов» → «Калькулятор».
    """
    use = list(df.columns) if with_class else list(df.columns[1:])
    return [[f"{c}={v}" for c, v in zip(use, row)] for row in df[use].itertuples(index=False)]


def demo_loginom():
    """Пункт 1: демо-пример Loginom — CLOPE с r = 2,6 на Mushroom (8124 гриба, класс и 22 признака).

    Решение строится с классом в транзакциях (как в Loginom) и без класса.
    «Грязный» кластер содержит и съедобные, и ядовитые грибы.
    """
    mush = pd.read_csv(DATA / "agaricus-lepiota.data", header=None, names=["class"] + [f"a{i}" for i in range(1, 23)])
    tr = mushroom_transactions(mush)
    lab, prof, hist, it = clope(tr, r=2.6)
    lab_nc = clope(mushroom_transactions(mush, with_class=False), r=2.6)[0]
    ct, ct_nc = (pd.crosstab(pd.Series(x, name="cluster"), mush["class"]) for x in (lab, lab_nc))
    dirty, dirty_nc = (int(((c > 0).sum(axis=1) == 2).sum()) for c in (ct, ct_nc))
    m_nc = external_metrics(mush["class"], lab_nc)
    n_items = len({i for t in tr for i in t})
    print(f"Транзакций: {len(tr)}, уникальных элементов: {n_items}")
    print(f"С классом в транзакции: кластеров {len(ct)}, «грязных» {dirty}, Profit = {prof:.4f}, итераций {it}")
    print(f"Без класса: кластеров {len(ct_nc)}, «грязных» {dirty_nc}, Purity = {m_nc['Purity']:.4f}, ARI = {m_nc['ARI']:.4f}")
    SUMMARY["demo"] = {"n_tr": len(tr), "n_items": n_items, "k": int(len(ct)), "dirty": dirty, "profit": prof,
                       "iter": it, "k_nc": int(len(ct_nc)), "dirty_nc": dirty_nc, "purity_nc": m_nc["Purity"],
                       "ari_nc": m_nc["ARI"], "dirty_nc_sizes": ct_nc[(ct_nc > 0).sum(axis=1) == 2].to_dict("index")}
    return pd.DataFrame(cluster_stats(tr, lab)).sort_values("S", ascending=False)


def loginom_comparison(st_mush):
    """Сопоставляет пять самых мощных кластеров с таблицей «Параметры кластеров» демо-примера Loginom."""
    ref = pd.DataFrame({"N": [1728, 1296, 1296, 768, 512], "W": [35, 35, 35, 34, 33],
                        "S": [39744, 29808, 29808, 17664, 11776]})
    own = st_mush.head(5)[["N", "W", "S"]].reset_index(drop=True)
    tab = pd.concat([ref.add_suffix(" (Loginom)"), own.add_suffix(" (Python)")], axis=1)
    tab.insert(0, "Ранг по S", range(1, 6))
    tab.to_csv(RES / "demo_mushroom.csv", index=False)
    return tab


def toy_example():
    """Пример из статьи Loginom: два разбиения транзакций {ab, abc, acd, de, def} и их Profit при r = 2."""
    toy = [list("ab"), list("abc"), list("acd"), list("de"), list("def")]
    for name, labels in [("(1)", [0, 0, 0, 1, 1]), ("(2)", [0, 0, 1, 1, 1])]:
        st = pd.DataFrame(cluster_stats(toy, labels))
        print(name, st[["N", "W", "S", "H", "G"]].round(3).to_dict("records"),
              "Profit(r=2) =", round(sum(s["S"] * s["N"] / s["W"] ** 2 for _, s in st.iterrows()) / 5, 3))


NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
DESCRIPTION = {
    "age": "Возраст, лет", "sex": "Пол", "cp": "Тип боли в груди",
    "trestbps": "Артериальное давление в покое, мм рт. ст.", "chol": "Холестерин сыворотки, мг/дл",
    "fbs": "Сахар крови натощак > 120 мг/дл", "restecg": "ЭКГ в покое",
    "thalach": "Максимальная ЧСС при нагрузке, уд/мин", "exang": "Стенокардия, вызванная нагрузкой",
    "oldpeak": "Депрессия сегмента ST при нагрузке, мм", "slope": "Наклон сегмента ST на пике нагрузки",
    "ca": "Число сосудов, окрашенных при флюороскопии", "thal": "Таллиевый стресс-тест",
    "target": "Диагноз (сужение артерии > 50 %)",
}
CODES = {
    "sex": {0: "жен.", 1: "муж."},
    "cp": {1: "типичная стенокардия", 2: "атипичная стенокардия", 3: "неангинозная боль", 4: "бессимптомно"},
    "fbs": {0: "нет", 1: "да"},
    "restecg": {0: "норма", 1: "аномалия ST-T", 2: "гипертрофия ЛЖ"},
    "exang": {0: "нет", 1: "да"},
    "slope": {1: "восходящий", 2: "плоский", 3: "нисходящий"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {3: "норма", 6: "фикс. дефект", 7: "обрат. дефект"},
    "target": {0: "нет болезни", 1: "болезнь"},
}


def describe_dataset():
    """Пункт 2: загружает processed.cleveland.data, удаляет строки с пропусками («?»), описывает признаки.

    target = 1 при num > 0 (наличие болезни) — только внешний эталон для оценки кластеров.
    Возвращает (df, feat_tab).
    """
    raw = pd.read_csv(DATA / "processed.cleveland.data", header=None, names=FEATURES + ["num"], na_values="?")
    miss = raw.isna().sum()
    df = raw.dropna().reset_index(drop=True)
    for c in CATEGORICAL + ["num"]:
        df[c] = df[c].astype(int)
    df["target"] = (df["num"] > 0).astype(int)
    SUMMARY["data"] = {"n_raw": len(raw), "n_missing": int(raw.isna().any(axis=1).sum()),
                       "miss_cols": miss[miss > 0].to_dict(), "n": len(df),
                       "num_dist": df["num"].value_counts().sort_index().to_dict(),
                       "target_dist": df["target"].value_counts().sort_index().to_dict()}
    s = SUMMARY["data"]
    print(f"Исходно записей: {s['n_raw']}, с пропусками: {s['n_missing']} ({s['miss_cols']}), после удаления: {s['n']}")
    print("Распределение num:", s["num_dist"])
    print("Распределение target:", s["target_dist"])
    feat_tab = pd.DataFrame({
        "Признак": FEATURES,
        "Описание": [DESCRIPTION[c] for c in FEATURES],
        "Тип": ["количественный" if c in NUMERIC else ("порядковый" if c == "ca" else "номинальный") for c in FEATURES],
        "Значения": [f"{df[c].min():g}–{df[c].max():g}" if c in NUMERIC else
                     "; ".join(f"{k} — {v}" for k, v in CODES[c].items()) if c != "ca" else "0, 1, 2, 3" for c in FEATURES],
    })
    feat_tab.to_csv(RES / "features.csv", index=False)
    return df, feat_tab


def descriptive_numeric(df):
    """Пункт 3: описательные статистики количественных признаков, включая асимметрию и эксцесс."""
    desc = df[NUMERIC].describe().T.assign(skew=df[NUMERIC].skew(), kurt=df[NUMERIC].kurt())
    desc.T.round(2).to_csv(RES / "desc_numeric.csv")
    return desc


def categorical_frequencies(df):
    """Таблица абсолютных и относительных частот категорий категориальных признаков и диагноза."""
    freq = pd.DataFrame([{"Признак": c, "Код": k, "Категория": CODES[c][k], "n": v, "%": 100 * v / len(df)}
                         for c in CATEGORICAL + ["target"] for k, v in df[c].value_counts().sort_index().items()])
    freq.round(1).to_csv(RES / "freq_categorical.csv", index=False)
    return freq


def plot_numeric(df):
    """Гистограммы и диаграммы размаха количественных признаков в разрезе диагноза."""
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5))
    for j, c in enumerate(NUMERIC):
        for t, lab in CODES["target"].items():
            axes[0, j].hist(df.loc[df.target == t, c], bins=15, alpha=0.6, label=lab)
        axes[0, j].set_title(DESCRIPTION[c].split(",")[0])
        sns.boxplot(x=df["target"].map(CODES["target"]), y=df[c], ax=axes[1, j])
        axes[1, j].set(xlabel="", ylabel=c)
    axes[0, 0].legend()
    plt.tight_layout()
    savefig("fig_numeric")


def plot_categorical(df):
    """Столбчатые диаграммы частот категорий каждого категориального признака с разбивкой по диагнозу."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, c in zip(axes.ravel(), CATEGORICAL):
        ct = pd.crosstab(df[c].map(CODES[c]), df["target"].map(CODES["target"]))
        ct.loc[[CODES[c][k] for k in sorted(df[c].unique())]].plot(kind="bar", ax=ax, rot=20)
        ax.set(title=DESCRIPTION[c], xlabel="")
        ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("fig_categorical")


BINS = {
    "age": ([0, 49, 59, 200], ["<50", "50-59", ">=60"]),
    "trestbps": ([0, 119, 139, 1000], ["<120", "120-139", ">=140"]),
    "chol": ([0, 199, 239, 1000], ["<200", "200-239", ">=240"]),
    "oldpeak": ([-1, 0, 2, 100], ["0", "0-2", ">2"]),
}
LEVELS = ["низкая", "средняя", "высокая"]


def discretize(df):
    """Переводит количественные признаки в 3 категории: age, trestbps, chol, oldpeak — по клиническим
    порогам BINS, thalach — по терцилям выборки. Возвращает (cat_df, thalach_edges).
    """
    out = df[FEATURES].astype(str)
    for c, (edges, labels) in BINS.items():
        out[c] = pd.cut(df[c], edges, labels=labels).astype(str)
    thalach, edges = pd.qcut(df["thalach"], 3, labels=LEVELS, retbins=True)
    out["thalach"] = thalach.astype(str)
    return out, edges


def cramers_v(x, y):
    """Критерий хи-квадрат и коэффициент Крамера V = sqrt(chi2 / (n * (min(r, c) - 1))); возвращает (chi2, p, V)."""
    ct = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    return chi2, p, np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))


def association_analysis(df, cat_all):
    """Связь признаков с диагнозом (V Крамера) и тепловые карты: Спирмен для количественных, V Крамера для всех."""
    assoc = pd.DataFrame([{"feature": c, "V": v, "p": p} for c in FEATURES
                          for _, p, v in [cramers_v(cat_all[c], df["target"])]]).sort_values("V", ascending=False)
    assoc.to_csv(RES / "assoc_target.csv", index=False)
    vmat = pd.DataFrame([[1.0 if a == b else cramers_v(cat_all[a], cat_all[b])[2] for b in FEATURES]
                         for a in FEATURES], index=FEATURES, columns=FEATURES)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1, 1.6]})
    sns.heatmap(df[NUMERIC].corr(method="spearman"), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Корреляция Спирмена (количественные признаки)")
    sns.heatmap(vmat, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=axes[1], annot_kws={"size": 8})
    axes[1].set_title("V Крамера (все признаки в категориальной форме)")
    plt.tight_layout()
    savefig("fig_corr")
    return assoc


def to_transactions(cat_df):
    """Каждая строка → транзакция из элементов вида «признак=значение»."""
    return [[f"{c}={v}" for c, v in zip(cat_df.columns, row)] for row in cat_df.itertuples(index=False)]


def to_long_table(cat_df):
    """Транзакционная таблица Loginom «Транзакция (ID) — Элемент (item)»: аналог узлов
    «Добавление ID» → «Свёртка столбцов» → «Калькулятор» (Concat).
    """
    long = cat_df.reset_index().melt(id_vars="index", var_name="feature", value_name="value")
    long["ID"] = long["index"] + 1
    long["item"] = long["feature"] + "=" + long["value"]
    return long.sort_values(["ID", "feature"])[["ID", "item"]].reset_index(drop=True)


def transform_data(cat_all, thalach_edges):
    """Пункт 4: таблица правил дискретизации, транзакции и транзакционная таблица для Loginom."""
    e = thalach_edges
    rules = {c: ("; ".join(labels), "клинические пороги") for c, (_, labels) in BINS.items()}
    rules["thalach"] = (f"низкая ≤ {e[1]:.0f}; средняя ≤ {e[2]:.0f}; высокая > {e[2]:.0f}", "терцили выборки")
    disc_tab = pd.DataFrame([{"Признак": c, "Категории": rules[c][0], "Основание": rules[c][1],
                              "Частоты": "; ".join(f"{k}: {v}" for k, v in cat_all[c].value_counts().items())}
                             for c in NUMERIC])
    disc_tab.to_csv(RES / "discretization.csv", index=False)
    transactions, long = to_transactions(cat_all), to_long_table(cat_all)
    long.to_csv(DATA / "heart_transactions_loginom.csv", index=False)
    long.head(16).to_csv(RES / "long_fragment.csv", index=False)
    n_items = len({i for t in transactions for i in t})
    SUMMARY["transform"] = {"n_items": n_items, "len_tr": len(transactions[0]), "n_long": len(long),
                            "thalach_edges": [float(x) for x in e]}
    print(f"Транзакций: {len(transactions)}, элементов в транзакции: {len(transactions[0])}, различных элементов: {n_items}")
    print(transactions[0])
    return transactions, disc_tab


def plot_discrete(df, cat_all):
    """Распределения категорий дискретизированных количественных признаков в разрезе диагноза."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, c in zip(axes, NUMERIC):
        order = BINS[c][1] if c in BINS else LEVELS
        pd.crosstab(cat_all[c], df["target"].map(CODES["target"])).loc[order].plot(kind="bar", ax=ax, rot=0)
        ax.set(title=c, xlabel="")
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("fig_discrete")


MIN_SHARE, N_ORDERS = 0.05, 50
R_GRID = np.round(np.arange(1.0, 4.001, 0.1), 1)


def hamming_matrix(cat_df):
    """Нормированное расстояние Хэмминга: d(x, y) = (1/p) * sum_j [x_j != y_j]."""
    codes = np.column_stack([pd.factorize(cat_df[c])[0] for c in cat_df.columns])
    return (codes[:, None, :] != codes[None, :, :]).mean(axis=2)


def silhouette(D, labels):
    """Средний коэффициент силуэта по предвычисленной матрице расстояний D."""
    return metrics.silhouette_score(D, labels, metric="precomputed") if len(set(labels)) > 1 else np.nan


def stability(solutions):
    """Устойчивость: средний попарный ARI между разбиениями из разных запусков."""
    return float(np.mean([metrics.adjusted_rand_score(a, b) for a, b in itertools.combinations(solutions, 2)]))


def merge_small_clusters(transactions, labels, r, min_share=MIN_SHARE):
    """Переназначает транзакции мелких кластеров (доля < min_share) в крупные.

    Каждая транзакция мелкого кластера переносится в тот крупный кластер, для
    которого DeltaAdd максимально (тот же критерий, что в CLOPE). Если крупных
    кластеров нет, метки не меняются. Результат перенумерован по убыванию размера.
    """
    labels = np.asarray(labels)
    sizes = np.bincount(labels)
    big = [k for k in range(len(sizes)) if sizes[k] >= min_share * len(labels)]
    if not big:
        return labels
    clusters = {k: c for k, c in build_clusters(transactions, labels).items() if k in big}
    out = [lab if lab in clusters else max(big, key=lambda k: delta_add(clusters[k], t, r))
           for t, lab in zip(transactions, labels)]
    remap = {old: new for new, (old, _) in enumerate(Counter(out).most_common())}
    return np.array([remap[lab] for lab in out])


def clope_grid_search(transactions, cat_all, y):
    """Пункт 5.1: перебор r из R_GRID, для каждого r — N_ORDERS случайных порядков транзакций.

    Для лучшего по Profit решения: число кластеров, число крупных кластеров, покрытие,
    силуэт по Хэммингу, устойчивость к порядку и (только для контроля) ARI и NMI
    с диагнозом. Возвращает (grid, best_by_r, H_mat).
    """
    n, H_mat = len(transactions), hamming_matrix(cat_all)
    grid, best_by_r = [], {}
    for r in R_GRID:
        runs = [clope(transactions, r=r, seed=s) for s in range(N_ORDERS)]
        best = max(runs, key=lambda x: x[1])
        sizes = np.bincount(best[0])
        merged = merge_small_clusters(transactions, best[0], r)
        ext = external_metrics(y, merged)
        grid.append({"r": r, "k": len(sizes), "k_large": int((sizes >= MIN_SHARE * n).sum()),
                     "coverage": sizes[sizes >= MIN_SHARE * n].sum() / n, "profit": best[1],
                     "sil_H": silhouette(H_mat, merged),
                     "stability": stability([merge_small_clusters(transactions, lab, r) for lab, *_ in runs]),
                     "ARI": ext["ARI"], "NMI": ext["NMI"]})
        best_by_r[r] = {"labels": np.array(best[0]), "merged": merged, "history": best[2], "n_iter": best[3]}
    grid = pd.DataFrame(grid)
    grid.round(4).to_csv(RES / "clope_grid.csv", index=False)
    return grid, best_by_r, H_mat


def choose_r(grid):
    """Выбор r*: не менее 3 крупных кластеров, максимум покрытия, при равенстве — максимум устойчивости.

    Строит графики характеристик решений CLOPE в зависимости от r (пунктир — r*).
    """
    r_star = float(grid[grid.k_large >= 3].sort_values(["coverage", "stability"], ascending=False).iloc[0]["r"])
    panels = [("Число кластеров", True, [("k", "o-", "все кластеры"), ("k_large", "s-", "крупные (≥ 5 %)")]),
              ("Profit (максимум из 50 запусков)", True, [("profit", "o-", None)]),
              ("Покрытие и устойчивость", False, [("coverage", "o-", "доля в крупных кластерах"),
                                                  ("stability", "s-", "устойчивость (ARI)")]),
              ("Качество решения", False, [("sil_H", "o-", "силуэт (Хэмминг)"), ("ARI", "s-", "ARI с диагнозом"),
                                           ("NMI", "^-", "NMI с диагнозом")])]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (title, log, lines) in zip(axes.ravel(), panels):
        for col, fmt, label in lines:
            ax.plot(grid.r, grid[col], fmt, label=label)
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
        if len(lines) > 1:
            ax.legend()
        ax.axvline(r_star, color="grey", ls="--")
        ax.set_xlabel("r")
    plt.tight_layout()
    savefig("fig_clope_grid")
    return r_star


def clope_final(transactions, best_by_r, r_star, y):
    """Пункт 5.2: итоговое решение CLOPE при r*, график Profit по итерациям и параметры кластеров.

    Возвращает (lab_raw, lab_clope, st_fin): метки до и после объединения мелких кластеров
    и таблицу N, W, S, H, G с долей больных.
    """
    sol = best_by_r[r_star]
    lab_raw, lab_clope = sol["labels"], sol["merged"]
    raw_sizes = np.bincount(lab_raw)
    print(f"r* = {r_star}: кластеров до объединения {len(raw_sizes)}, размеры {raw_sizes.tolist()}")
    print(f"после объединения мелких: {len(set(lab_clope))}, размеры {np.bincount(lab_clope).tolist()}, итераций {sol['n_iter']}")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(len(sol["history"])), sol["history"], "o-")
    ax.set(xlabel="Итерация (0 — после инициализации)", ylabel="Profit")
    savefig("fig_clope_history")
    pd.DataFrame(cluster_stats(transactions, lab_raw)).to_csv(RES / "clope_clusters_raw.csv", index=False)
    st_fin = pd.DataFrame(cluster_stats(transactions, lab_clope))
    st_fin["Доля болезни, %"] = [100 * y[lab_clope == k].mean() for k in st_fin.cluster]
    st_fin["cluster"] += 1
    st_fin.round(3).to_csv(RES / "clope_clusters.csv", index=False)
    SUMMARY["clope"] = {"r": r_star, "k_raw": int(len(raw_sizes)), "raw_sizes": raw_sizes.tolist(),
                        "k": len(set(lab_clope)), "sizes": np.bincount(lab_clope).tolist(), "n_iter": sol["n_iter"],
                        "history": sol["history"], "profit_raw": float(sol["history"][-1]),
                        "profit_merged": profit(list(build_clusters(transactions, lab_clope).values()), r_star)}
    return lab_raw, lab_clope, st_fin


def item_profile(transactions, labels):
    """Нормированная гистограмма CLOPE: доля транзакций кластера, содержащих элемент, Occ(i, C) / N(C)."""
    rows = {}
    for t, lab in zip(transactions, labels):
        for i in t:
            rows.setdefault(i, Counter())[lab] += 1
    prof = pd.DataFrame(rows).T.fillna(0)
    prof = prof[sorted(prof.columns)]
    return prof / pd.Series(labels).value_counts().sort_index().values


def clope_item_profiles(transactions, lab_clope):
    """Пункт 5.3: тепловая карта профиля элементов и характерные элементы кластеров (доля ≥ 0,5 и lift ≥ 1,3)."""
    prof = item_profile(transactions, lab_clope)
    prof.columns = [f"К{k + 1}" for k in prof.columns]
    overall = pd.Series({i: sum(i in t for t in transactions) / len(transactions) for i in prof.index})
    ordered = prof.loc[sorted(prof.index, key=lambda i: (FEATURES.index(i.split("=")[0]), i))]
    fig, ax = plt.subplots(figsize=(1.3 * prof.shape[1] + 4, 11))
    sns.heatmap(ordered.assign(**{"Вся выборка": overall[ordered.index]}), annot=True, fmt=".2f",
                cmap="YlOrRd", ax=ax, cbar_kws={"label": "Occ(i, C) / N(C)"}, annot_kws={"size": 9})
    ax.set_ylabel("Элемент")
    savefig("fig_clope_items")
    rows = []
    for k, col in enumerate(prof.columns):
        top = prof[(prof[col] >= 0.5) & (prof[col] / overall[prof.index] >= 1.3)][col].sort_values(ascending=False)
        rows.append({"Кластер": col, "N": int((lab_clope == k).sum()),
                     "Характерные элементы (доля в кластере)": "; ".join(f"{i} ({v:.2f})" for i, v in top.items())})
    char_tab = pd.DataFrame(rows)
    char_tab.to_csv(RES / "clope_char.csv", index=False)
    return char_tab


GOWER_NUMERIC = NUMERIC + ["ca"]
GOWER_NOMINAL = [c for c in CATEGORICAL if c != "ca"]


def gower_matrix(df, numeric=GOWER_NUMERIC, nominal=GOWER_NOMINAL):
    """Матрица расстояний Гауэра: d(i, j) = (1/p) * sum_f delta_ijf, где delta_ijf = |x_if - x_jf| / R_f
    для количественных и порядковых признаков (R_f — размах), delta_ijf = [x_if != x_jf] для номинальных.
    """
    total = 0
    for c in numeric + nominal:
        x = df[c].to_numpy().astype(float)
        total = total + (np.abs(x[:, None] - x[None, :]) / np.ptp(x) if c in numeric else x[:, None] != x[None, :])
    return total / len(numeric + nominal)


def evaluate_solution(labels, y, H_mat, G_mat, df, name):
    """Внешние метрики относительно диагноза, силуэт по Хэммингу и по Гауэру и кросс-таблица «кластер × диагноз»."""
    m = external_metrics(y, labels)
    m["Silhouette (Хэмминг)"] = silhouette(H_mat, labels)
    m["Silhouette (Гауэр)"] = silhouette(G_mat, labels)
    ct = pd.crosstab(pd.Series(labels + 1, name=name), pd.Series(df["target"].map(CODES["target"]), name="Диагноз"))
    show(pd.Series(m), 4)
    show(ct)
    return m, ct


def kruskal_eta2(x, labels):
    """Критерий Краскела–Уоллиса и размер эффекта eta^2_H = (H - k + 1) / (n - k); возвращает (H, p, eta2)."""
    groups = [x[labels == g] for g in np.unique(labels)]
    h, p = stats.kruskal(*groups)
    return h, p, (h - len(groups) + 1) / (len(x) - len(groups))


def feature_significance(cat_all, df, labels, file, fig_name=None):
    """Значимость признаков для разбиения, по убыванию V Крамера.

    Для всех признаков (в категориальной форме) — хи-квадрат и V Крамера; для исходных
    количественных — дополнительно Краскел–Уоллис и eta^2_H. Таблица сохраняется в
    results/<file>; при заданном fig_name строится столбчатая диаграмма V Крамера.
    """
    labels = np.asarray(labels)
    rows = []
    for c in FEATURES:
        chi2, p, v = cramers_v(cat_all[c].to_numpy(), labels)
        h, ph, e = kruskal_eta2(df[c].to_numpy(), labels) if c in NUMERIC else (np.nan,) * 3
        rows.append({"feature": c, "chi2": chi2, "p_chi2": p, "V": v, "H": h, "p_H": ph, "eta2": e,
                     "description": DESCRIPTION[c]})
    sig = pd.DataFrame(rows).sort_values("V", ascending=False).reset_index(drop=True)
    sig.round(4).to_csv(RES / file, index=False)
    if fig_name:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=sig, y="feature", x="V", ax=ax, color="tab:blue")
        ax.set(xlabel="V Крамера (признак × кластер)", ylabel="")
        savefig(fig_name)
    return sig


def gower_example(df, G_mat, i=0, j=1):
    """Пункт 6.1: среднее расстояние Гауэра и покомпонентные вклады delta_ijf в d(i, j)."""
    iu = np.triu_indices(len(df), 1)
    contrib = pd.DataFrame([(c, df[c].iloc[i], df[c].iloc[j],
                             abs(df[c].iloc[i] - df[c].iloc[j]) / (df[c].max() - df[c].min()) if c in GOWER_NUMERIC
                             else float(df[c].iloc[i] != df[c].iloc[j])) for c in GOWER_NUMERIC + GOWER_NOMINAL],
                           columns=["feature", "x_i", "x_j", "delta"])
    contrib.to_csv(RES / "gower_example.csv", index=False)
    SUMMARY["gower_example"] = {"d01": float(G_mat[i, j]), "mean": float(G_mat[iu].mean())}
    print("Размер матрицы:", G_mat.shape, " среднее расстояние:", round(G_mat[iu].mean(), 4))
    print(f"d({i}, {j}) =", round(G_mat[i, j], 4), "= среднее вкладов", round(contrib.delta.mean(), 4))
    return contrib


def pam(D, k, max_iter=100):
    """Алгоритм PAM (Partitioning Around Medoids, Kaufman & Rousseeuw).

    Минимизирует E = sum_i d(x_i, m_c(i)) — сумму расстояний объектов до ближайшего
    медоида. BUILD: жадный выбор k начальных медоидов. SWAP: перебор замен
    «медоид ↔ не медоид», принимается замена с наибольшим уменьшением E, пока такие
    замены существуют. Возвращает (labels, medoids, cost), кластеры — по убыванию размера.
    """
    n = D.shape[0]
    medoids = [int(np.argmin(D.sum(axis=1)))]
    while len(medoids) < k:
        nearest = D[:, medoids].min(axis=1)
        gains = [np.maximum(nearest - D[:, c], 0).sum() if c not in medoids else -1 for c in range(n)]
        medoids.append(int(np.argmax(gains)))
    cost = D[:, medoids].min(axis=1).sum()
    for _ in range(max_iter):
        best = (0.0, None, None)
        for mi, h in itertools.product(range(k), range(n)):
            if h not in medoids:
                cand = medoids[:mi] + [h] + medoids[mi + 1:]
                if (delta := D[:, cand].min(axis=1).sum() - cost) < best[0] - 1e-12:
                    best = (delta, mi, h)
        if best[1] is None:
            break
        medoids[best[1]] = best[2]
        cost += best[0]
    labels = np.argmin(D[:, medoids], axis=1)
    order = np.argsort(-np.bincount(labels, minlength=k))
    return np.argsort(order)[labels], [medoids[o] for o in order], float(cost)


def pam_grid_search(G_mat, y):
    """Пункт 6.2: PAM для k = 2..8 по матрице Гауэра; k выбирается по максимуму среднего силуэта.

    Возвращает (pam_grid, pam_sol, k_best).
    """
    sol = {k: pam(G_mat, k) for k in range(2, 9)}
    grid = pd.DataFrame([{"k": k, "cost": cost, "sil_G": silhouette(G_mat, lab), "min_share": np.bincount(lab).min() / len(lab),
                          **{m: external_metrics(y, lab)[m] for m in ("ARI", "NMI")}} for k, (lab, _, cost) in sol.items()])
    k_best = int(grid.loc[grid.sil_G.idxmax(), "k"])
    grid.round(4).to_csv(RES / "pam_grid.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(grid.k, grid.cost, "o-")
    axes[0].set_title("Целевая функция PAM (сумма расстояний до медоидов)")
    for col, fmt, label in [("sil_G", "o-", "силуэт (Гауэр)"), ("ARI", "s-", "ARI с диагнозом"), ("NMI", "^-", "NMI с диагнозом")]:
        axes[1].plot(grid.k, grid[col], fmt, label=label)
    axes[1].legend()
    axes[1].set_title("Качество разбиения")
    for ax in axes:
        ax.set_xlabel("k")
        ax.axvline(k_best, color="grey", ls="--")
    plt.tight_layout()
    savefig("fig_pam_grid")
    return grid, sol, k_best


def pam_profiles(df, G_mat, lab_pam, med_pam):
    """Пункт 6.3: силуэтная диаграмма, медоиды и профили кластеров PAM (средние и моды с долями)."""
    k = len(med_pam)
    sil_s = metrics.silhouette_samples(G_mat, lab_pam, metric="precomputed")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y0 = 0
    for g in range(k):
        v = np.sort(sil_s[lab_pam == g])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=0.8, label=f"Кластер {g + 1}")
        y0 += len(v) + 5
    ax.axvline(sil_s.mean(), color="red", ls="--", label=f"среднее = {sil_s.mean():.3f}")
    ax.set_xlabel("s(i)")
    ax.set_yticks([])
    ax.legend()
    savefig("fig_pam_silhouette")
    med_tab = df.loc[med_pam, FEATURES].T.astype(object)
    med_tab.columns = [f"Медоид К{g + 1}" for g in range(k)]
    for c in CATEGORICAL:
        med_tab.loc[c] = [CODES[c][int(v)] for v in med_tab.loc[c]]
    med_tab.to_csv(RES / "pam_medoids.csv")
    prof = {}
    for g in range(k):
        sub = df[lab_pam == g]
        modes = {c: sub[c].value_counts(normalize=True) for c in CATEGORICAL}
        prof[f"К{g + 1}"] = {**{c: f"{sub[c].mean():.1f}" if c in NUMERIC else
                                f"{CODES[c][int(modes[c].index[0])]} ({100 * modes[c].iloc[0]:.0f} %)" for c in FEATURES},
                             "N": len(sub), "Доля болезни, %": f"{100 * sub.target.mean():.1f}"}
    pam_prof = pd.DataFrame(prof)
    pam_prof.to_csv(RES / "pam_profile.csv")
    return pam_prof


def standardize(df):
    """z-преобразование количественных признаков: z = (x - mean) / std."""
    return (df[NUMERIC] - df[NUMERIC].mean()) / df[NUMERIC].std()


def pam_heatmaps(df, cat_all, lab_pam):
    """Тепловые карты профилей PAM: средние z-значения и отклонения долей категорий от средних по выборке."""
    num_prof = standardize(df).groupby(lab_pam).mean()
    num_prof.index = [f"К{k + 1}" for k in num_prof.index]
    dummies = pd.get_dummies(cat_all[CATEGORICAL].apply(lambda s: s.name + "=" + s))
    cat_prof = (dummies.groupby(lab_pam).mean() - dummies.mean()).set_axis(num_prof.index)
    fig, axes = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 3.5]})
    sns.heatmap(num_prof.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[0])
    axes[0].set_title("Количественные: среднее z-значение")
    sns.heatmap(cat_prof.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[1], annot_kws={"size": 8})
    axes[1].set_title("Категориальные: отклонение доли категории от средней по выборке")
    plt.tight_layout()
    savefig("fig_pam_profile")


def clope_best_of(transactions, r, n_orders=50):
    """Запускает CLOPE для n_orders случайных порядков и возвращает метки решения с максимальным Profit."""
    return max((clope(transactions, r=r, seed=s) for s in range(n_orders)), key=lambda x: x[1])[0]


def bootstrap_stability(transactions, G_mat, lab_clope, lab_pam, r_star, k_pam, n_rep=20, share=0.8):
    """Устойчивость к составу выборки: n_rep подвыборок по 80 %, ARI с исходным разбиением тех же объектов."""
    n = len(transactions)
    ari = {"CLOPE": [], "Гауэр + PAM": []}
    for s in range(n_rep):
        idx = np.sort(np.random.default_rng(s).choice(n, int(share * n), replace=False))
        tr_sub = [transactions[i] for i in idx]
        lab_c = merge_small_clusters(tr_sub, clope_best_of(tr_sub, r_star, n_orders=10), r_star)
        ari["CLOPE"].append(metrics.adjusted_rand_score(lab_clope[idx], lab_c))
        ari["Гауэр + PAM"].append(metrics.adjusted_rand_score(lab_pam[idx], pam(G_mat[np.ix_(idx, idx)], k_pam)[0]))
    tab = pd.DataFrame({m: [np.mean(v), np.std(v)] for m, v in ari.items()}, index=["ARI среднее", "ARI ст. откл."])
    tab.round(3).to_csv(RES / "bootstrap.csv")
    SUMMARY["boot"] = tab.to_dict()
    return tab


def plot_significance(sig_clope, sig_pam):
    """Сравнительная диаграмма V Крамера признаков для разбиений CLOPE и PAM."""
    both = pd.DataFrame({"CLOPE": sig_clope.set_index("feature")["V"], "Гауэр + PAM": sig_pam.set_index("feature")["V"]})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    both.loc[sig_clope["feature"]].sort_values("Гауэр + PAM").plot(kind="barh", ax=ax)
    ax.set(xlabel="V Крамера (признак × кластер)", ylabel="")
    savefig("fig_sig_compare")


def kprototypes_check(df, k, y, H_mat, G_mat, lab_pam):
    """Контрольное разбиение k-prototypes (Huang): z-значения количественных и коды категориальных признаков."""
    X = pd.concat([standardize(df), df[CATEGORICAL].astype(str)], axis=1).to_numpy(dtype=object)
    kp = KPrototypes(n_clusters=k, init="Cao", n_init=10, random_state=0)
    lab_kp = kp.fit_predict(X, categorical=list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL))))
    ari_pam = metrics.adjusted_rand_score(lab_pam, lab_kp)
    print("gamma =", round(kp.gamma, 3), " размеры:", np.bincount(lab_kp).tolist())
    print("ARI(k-prototypes, PAM) =", round(ari_pam, 4))
    m_kp, _ = evaluate_solution(lab_kp, y, H_mat, G_mat, df, "Кластер k-prototypes")
    SUMMARY["kproto"] = {"k": k, "gamma": float(kp.gamma), "sizes": np.bincount(lab_kp).tolist(),
                         "metrics": m_kp, "ari_pam": ari_pam}
    return lab_kp, m_kp


def compare_solutions(m_clope, m_pam, m_kp, k_clope, k_pam, lab_clope, lab_pam):
    """Пункт 7: сводная таблица метрик методов и кросс-таблица «кластер CLOPE × кластер PAM»."""
    cmp = pd.DataFrame({"CLOPE": m_clope, "Гауэр + PAM": m_pam, "k-prototypes": m_kp}).T
    cmp.insert(0, "k", [k_clope, k_pam, k_pam])
    cmp.round(4).to_csv(RES / "compare.csv")
    ct = pd.crosstab(pd.Series(lab_clope + 1, name="CLOPE"), pd.Series(lab_pam + 1, name="PAM"))
    ct.to_csv(RES / "crosstab_clope_pam.csv")
    SUMMARY["compare"] = {"ari_clope_pam": metrics.adjusted_rand_score(lab_clope, lab_pam),
                          "nmi_clope_pam": metrics.normalized_mutual_info_score(lab_clope, lab_pam),
                          "ct": ct.to_dict("index")}
    print("ARI(CLOPE, PAM) =", round(SUMMARY["compare"]["ari_clope_pam"], 4),
          " NMI(CLOPE, PAM) =", round(SUMMARY["compare"]["nmi_clope_pam"], 4))
    show(ct)
    return cmp


def plot_comparison(G_mat, y, lab_clope, lab_pam, med_pam, r_star, k_pam, cmp):
    """Проекция MDS по матрице Гауэра с раскраской по диагнозу, CLOPE и PAM; диаграмма метрик методов."""
    emb = MDS(n_components=2, metric="precomputed", random_state=0, n_init=4, init="random").fit_transform(G_mat)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = [("Диагноз (эталон)", y), (f"CLOPE, r = {r_star}", lab_clope), (f"Гауэр + PAM, k = {k_pam}", lab_pam)]
    for ax, (title, lab) in zip(axes, panels):
        for g in np.unique(lab):
            name = CODES["target"][g] if title.startswith("Диагноз") else f"Кластер {g + 1}"
            ax.scatter(emb[lab == g, 0], emb[lab == g, 1], s=18, alpha=0.8, label=name)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[2].scatter(emb[med_pam, 0], emb[med_pam, 1], marker="*", s=300, c="black", label="медоиды")
    axes[2].legend(fontsize=9)
    plt.tight_layout()
    savefig("fig_mds")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmp[["ARI", "NMI", "V-measure", "Purity", "Silhouette (Гауэр)", "Silhouette (Хэмминг)"]].T.plot(kind="bar", ax=ax, rot=15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Значение метрики")
    savefig("fig_compare")


def to_py(o):
    """Рекурсивно приводит типы NumPy к встроенным типам Python для записи в JSON."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    return o.item() if isinstance(o, np.generic) else o


def save_results(lab_clope, lab_raw, lab_pam, lab_kp, y):
    """Сохраняет метки кластеров всех методов (results/labels.csv) и сводку ключевых чисел (results/summary.json)."""
    pd.DataFrame({"clope": lab_clope + 1, "clope_raw": lab_raw + 1, "pam": lab_pam + 1, "kproto": lab_kp + 1,
                  "target": y}).to_csv(RES / "labels.csv", index=False)
    (RES / "summary.json").write_text(json.dumps(to_py(SUMMARY), ensure_ascii=False, indent=1))
    print("Сохранено:", sorted(p.name for p in RES.iterdir()))


def main():
    """Выполняет все пункты лабораторной работы по порядку."""
    st_mush = demo_loginom()
    show(st_mush.head(5))
    show(loginom_comparison(st_mush))
    toy_example()
    df, feat_tab = describe_dataset()
    y, n = df["target"].to_numpy(), len(df)
    show(feat_tab)
    show(descriptive_numeric(df), 2)
    show(categorical_frequencies(df), 1)
    plot_numeric(df)
    plot_categorical(df)
    cat_all, thalach_edges = discretize(df)
    show(association_analysis(df, cat_all), 3)
    transactions, disc_tab = transform_data(cat_all, thalach_edges)
    show(disc_tab)
    plot_discrete(df, cat_all)
    grid, best_by_r, H_mat = clope_grid_search(transactions, cat_all, y)
    show(grid, 3)
    R_STAR = choose_r(grid)
    print("Выбранное r* =", R_STAR)
    lab_raw, lab_clope, st_fin = clope_final(transactions, best_by_r, R_STAR, y)
    K_CLOPE = len(set(lab_clope))
    show(st_fin, 3)
    show(clope_item_profiles(transactions, lab_clope))
    G_mat = gower_matrix(df)
    m_clope, ct_clope = evaluate_solution(lab_clope, y, H_mat, G_mat, df, "Кластер CLOPE")
    SUMMARY["clope"].update(metrics=m_clope, ct=ct_clope.to_dict("index"))
    sig_clope = feature_significance(cat_all, df, lab_clope, "clope_significance.csv", "fig_clope_sig")
    show(sig_clope, 4)
    show(gower_example(df, G_mat), 4)
    pam_grid, pam_sol, K_PAM = pam_grid_search(G_mat, y)
    print("Выбрано k =", K_PAM)
    show(pam_grid, 4)
    lab_pam, med_pam, cost_pam = pam_sol[K_PAM]
    show(pam_profiles(df, G_mat, lab_pam, med_pam))
    pam_heatmaps(df, cat_all, lab_pam)
    m_pam, ct_pam = evaluate_solution(lab_pam, y, H_mat, G_mat, df, "Кластер PAM")
    show(bootstrap_stability(transactions, G_mat, lab_clope, lab_pam, R_STAR, K_PAM), 3)
    sig_pam = feature_significance(cat_all, df, lab_pam, "pam_significance.csv")
    plot_significance(sig_clope, sig_pam)
    SUMMARY["pam"] = {"k": K_PAM, "sizes": np.bincount(lab_pam).tolist(), "cost": cost_pam,
                      "medoids": [int(m) for m in med_pam], "metrics": m_pam, "ct": ct_pam.to_dict("index")}
    show(sig_pam, 4)
    lab_kp, m_kp = kprototypes_check(df, K_PAM, y, H_mat, G_mat, lab_pam)
    cmp = compare_solutions(m_clope, m_pam, m_kp, K_CLOPE, K_PAM, lab_clope, lab_pam)
    show(cmp, 4)
    plot_comparison(G_mat, y, lab_clope, lab_pam, med_pam, R_STAR, K_PAM, cmp)
    save_results(lab_clope, lab_raw, lab_pam, lab_kp, y)


if __name__ == "__main__":
    main()
