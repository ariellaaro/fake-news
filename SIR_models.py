''' ================================
additional libraries
================================ '''
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from collections import defaultdict

''' ================================
dataset path (change as needed)
================================ '''

# the following runs on Google Colab
from google.colab import drive
drive.mount('/content/drive')

csv_path = "/content/drive/MyDrive/IME/MC_Fake_dataset.csv"
df = pd.read_csv(csv_path)

''' ================================
column formatting
================================ '''

# Format "labels" from 0/1 to fake/true
df["labels"] = df["labels"].map({0: "true", 1: "fake"})

# Format "retweet_relations" and "reply_relations" strings to tuples
def parse_relations(rel_str):
    if not isinstance(rel_str, str) or rel_str.strip() == "":
        return None
    relations = []
    parts = rel_str.split(",")
    for part in parts:
        items = part.strip().split("-")
        if len(items) >= 4:
            src = items[-2]
            dst = items[-1]
            relations.append((src, dst))
    return relations if relations else None

for col in ["retweet_relations", "reply_relations"]:
    if col in df.columns:
        df[col] = df[col].apply(parse_relations)

print("labels:\n", df["labels"].unique())
print("\nretweet_relations:")
print(df["retweet_relations"].dropna().head(1).tolist())
print("\nreply_relations:")
print(df["reply_relations"].dropna().head(1).tolist())

''' ================================
graph building
================================ '''

def build_graph(subdf):
    G = nx.DiGraph()
    for col in ["retweet_relations", "reply_relations"]:
        if col in subdf.columns:
            for rels in subdf[col].dropna():
                for src, dst in rels:
                    G.add_edge(src, dst)
    return G

G_fake = build_graph(df[df["labels"] == "fake"])
G_true = build_graph(df[df["labels"] == "true"])

''' ================================
SIR models
================================ '''

def draw_combined_graph(G_true, G_fake, title="Rede combinada", max_nodes=300, layout="spring"):
    """
    Desenha uma única rede com nós de duas origens (TRUE/FALSE) coloridos diferentes.
    """
    # Cria cópias para não alterar os originais
    Gt = G_true.copy()
    Gf = G_fake.copy()

    # Adiciona atributo indicando tipo
    nx.set_node_attributes(Gt, "TRUE", "tipo")
    nx.set_node_attributes(Gf, "FAKE", "tipo")

    # Combina os dois grafos
    G = nx.compose(Gt, Gf)

    # Limita o número de nós (se necessário)
    if G.number_of_nodes() > max_nodes:
        nodes_sample = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes_sample).copy()
        print(f"[{title}] mostrando {len(G.nodes())} nós de {G.number_of_nodes()}")

    # Escolhe layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.2, iterations=30, seed=42)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G)

    # Divide nós conforme tipo
    true_nodes = [n for n, d in G.nodes(data=True) if d.get("tipo") == "TRUE"]
    fake_nodes = [n for n, d in G.nodes(data=True) if d.get("tipo") == "FAKE"]

    # Desenha
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=true_nodes, node_color="steelblue", label="TRUE", node_size=40, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=fake_nodes, node_color="indianred", label="FAKE", node_size=40, alpha=0.8)

    plt.title(title)
    plt.legend(scatterpoints=1)
    plt.axis("off")
    plt.show()

# Exemplo de uso:
draw_combined_graph(G_true, G_fake, title="Rede de difusão combinada (TRUE vs FAKE)", max_nodes=5000)

# Filtra apenas as linhas com labels válidos
df = df[df["labels"].isin(["fake", "true"])].copy()

# Determina quantos pegar de cada
n_total =8000
n_each = n_total // 2

# Amostra aleatória estratificada (50/50 fake/true)
df_subset = pd.concat([
    df[df["labels"] == "fake"].sample(n=n_each, random_state=42),
    df[df["labels"] == "true"].sample(n=n_each, random_state=42)
])

print(df_subset["labels"].value_counts())

# Estados:
# 0 = S (susceptível)
# 1 = I_A (infectado por A, ex.: FAKE)
# 2 = R_A (recuperado de A)
# 3 = I_B (infectado por B, ex.: TRUE)
# 4 = R_B (recuperado de B)

def simulate_competitive_SIR(
    G,
    seeds_A=None,
    seeds_B=None,
    beta_A=0.05,
    beta_B=0.05,
    gamma_A=0.2,
    gamma_B=0.2,
    max_steps=200,
    conflict_rule="probabilistic",  # "probabilistic" | "priority_A" | "priority_B" | "higher_beta"
    edge_weight_attr=None,          # nome do atributo de peso (ex.: 'weight'); se None, usa 1.0
    rng_seed=42,
):
    """
    SIR competitivo em rede (passos discretos).
    Prob. de infecção em u: p = 1 - ∏(1 - beta*w) sobre vizinhos infecciosos.
    Recuperação: prob gamma por passo.
    Conflito A vs B no mesmo passo:
      - 'probabilistic' -> A com pA/(pA+pB) (e testa ocorrência do vencedor)
      - 'priority_A'    -> prioriza A
      - 'priority_B'    -> prioriza B
      - 'higher_beta'   -> vence maior beta; empate vira probabilistic
    Retorna:
      history: lista de dict {node: state} em cada passo (0 = inicial)
      series: dict com séries temporais S, I_A, R_A, I_B, R_B
    """
    rng = np.random.default_rng(rng_seed)
    states = {u: 0 for u in G.nodes()}

    seeds_A = set(seeds_A or [])
    seeds_B = set(seeds_B or [])
    overlap = seeds_A & seeds_B

    # Resolve semente em comum
    for u in overlap:
        if conflict_rule == "priority_A":
            states[u] = 1
        elif conflict_rule == "priority_B":
            states[u] = 3
        elif conflict_rule == "higher_beta":
            if beta_A > beta_B:
                states[u] = 1
            elif beta_B > beta_A:
                states[u] = 3
            else:
                states[u] = 1 if rng.random() < 0.5 else 3
        else:
            # probabilistic
            states[u] = 1 if rng.random() < 0.5 else 3

    for u in seeds_A - overlap:
        states[u] = 1
    for u in seeds_B - overlap:
        states[u] = 3

    def count_states(sts):
        cnt = defaultdict(int)
        for v in sts.values():
            cnt[v] += 1
        return cnt

    history = [states.copy()]
    cnt0 = count_states(states)
    series = {
        "S": [cnt0.get(0, 0)], "I_A": [cnt0.get(1, 0)], "R_A": [cnt0.get(2, 0)],
        "I_B": [cnt0.get(3, 0)], "R_B": [cnt0.get(4, 0)]
    }

    neighbors = {u: list(G.neighbors(u)) for u in G.nodes()}

    for t in range(1, max_steps + 1):
        new_states = states.copy()

        # Recuperação
        for u, st in states.items():
            if st == 1 and rng.random() < gamma_A:
                new_states[u] = 2
            elif st == 3 and rng.random() < gamma_B:
                new_states[u] = 4

        # Cálculo de infecções candidatas
        candidates_A = {}
        candidates_B = {}

        for u, st in states.items():
            if st != 0:  # só S podem ser infectados
                continue

            pA_keep = 1.0
            pB_keep = 1.0
            for v in neighbors[u]:
                w = 1.0
                if edge_weight_attr is not None:
                    w = G[u][v].get(edge_weight_attr, 1.0) or 1.0
                if states[v] == 1:
                    pA_keep *= (1.0 - beta_A * w)
                elif states[v] == 3:
                    pB_keep *= (1.0 - beta_B * w)

            pA = 1.0 - pA_keep
            pB = 1.0 - pB_keep
            if pA > 0:
                candidates_A[u] = pA
            if pB > 0:
                candidates_B[u] = pB

        # Resolve conflitos e aplica novas infecções
        targets = set(candidates_A) | set(candidates_B)
        for u in targets:
            pA = candidates_A.get(u, 0.0)
            pB = candidates_B.get(u, 0.0)

            if pA > 0 and pB == 0:
                if rng.random() < pA and new_states[u] == 0:
                    new_states[u] = 1
            elif pB > 0 and pA == 0:
                if rng.random() < pB and new_states[u] == 0:
                    new_states[u] = 3
            elif pA > 0 and pB > 0:
                if conflict_rule == "priority_A":
                    if rng.random() < pA and new_states[u] == 0:
                        new_states[u] = 1
                elif conflict_rule == "priority_B":
                    if rng.random() < pB and new_states[u] == 0:
                        new_states[u] = 3
                elif conflict_rule == "higher_beta":
                    if beta_A > beta_B:
                        if rng.random() < pA and new_states[u] == 0:
                            new_states[u] = 1
                    elif beta_B > beta_A:
                        if rng.random() < pB and new_states[u] == 0:
                            new_states[u] = 3
                    else:
                        prob_A = pA / (pA + pB)
                        winner_is_A = (rng.random() < prob_A)
                        if winner_is_A:
                            if rng.random() < pA and new_states[u] == 0:
                                new_states[u] = 1
                        else:
                            if rng.random() < pB and new_states[u] == 0:
                                new_states[u] = 3
                else:
                    # probabilistic (default)
                    prob_A = pA / (pA + pB)
                    winner_is_A = (rng.random() < prob_A)
                    if winner_is_A:
                        if rng.random() < pA and new_states[u] == 0:
                            new_states[u] = 1
                    else:
                        if rng.random() < pB and new_states[u] == 0:
                            new_states[u] = 3

        states = new_states
        history.append(states.copy())
        cnt = count_states(states)
        series["S"].append(cnt.get(0, 0))
        series["I_A"].append(cnt.get(1, 0))
        series["R_A"].append(cnt.get(2, 0))
        series["I_B"].append(cnt.get(3, 0))
        series["R_B"].append(cnt.get(4, 0))

        # Parada: acabou infecciosos
        if series["I_A"][-1] == 0 and series["I_B"][-1] == 0:
            break

    return history, series

# ========================================
# PASSO 2 — Simulação competitiva SIR
# ========================================

import numpy as np
import networkx as nx

# --- Parâmetros da simulação ---
beta_A, beta_B = 0.06, 0.05   # taxas de transmissão
gamma_A, gamma_B = 0.25, 0.25 # taxas de recuperação
max_steps = 300
rng = np.random.default_rng(42)

# --- Constrói a rede com os mesmos índices do df_subset ---
# alvo: grau médio ≈ 8  → p = 8 / (n-1)
n = len(df_subset)
target_k = 8
p = target_k / max(n - 1, 1)

# Cria grafo Erdős–Rényi
G_tmp = nx.erdos_renyi_graph(n=n, p=p, seed=42)

# Relabela nós 0..n-1 com os índices reais do df_subset
mapping = {i: idx for i, idx in enumerate(df_subset.index)}
G = nx.relabel_nodes(G_tmp, mapping)

print(f"Grafo criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")

# --- Seleciona sementes iniciais (10 de cada tipo) ---
idx_fake = df_subset.index[df_subset["labels"] == "fake"].to_list()
idx_true = df_subset.index[df_subset["labels"] == "true"].to_list()

# Escolhe aleatoriamente 10 de cada
seeds_A = rng.choice(idx_fake, size=10, replace=False).tolist()
seeds_B = rng.choice(idx_true, size=10, replace=False).tolist()

print(f"Sementes FAKE: {seeds_A[:5]} ...")
print(f"Sementes TRUE: {seeds_B[:5]} ...")

# --- Executa o modelo competitivo SIR ---
history, series = simulate_competitive_SIR(
    G,
    seeds_A=seeds_A,
    seeds_B=seeds_B,
    beta_A=beta_A, beta_B=beta_B,
    gamma_A=gamma_A, gamma_B=gamma_B,
    max_steps=max_steps,
    conflict_rule="probabilistic",  # regra de competição entre A e B
    edge_weight_attr=None,
    rng_seed=2025
)

print(f"Simulação concluída em {len(series['S']) - 1} passos.")

# =========================================
# FIGURA 1 — Evolução temporal da difusão (com recuperados)
# =========================================
plt.figure(figsize=(9, 5))

# Infectados
plt.plot(series["I_A"], color="red", linewidth=2.2, label="I_A — Infectados FAKE")
plt.plot(series["I_B"], color="blue", linewidth=2.2, label="I_B — Infectados TRUE")

# Recuperados (novas curvas)
plt.plot(series["R_A"], color="salmon", linewidth=2, linestyle=":", label="R_A — Recuperados FAKE")
plt.plot(series["R_B"], color="skyblue", linewidth=2, linestyle=":", label="R_B — Recuperados TRUE")

# Suscetíveis
plt.plot(series["S"], color="gray", linestyle="--", linewidth=2, label="S — Susceptíveis")

plt.xlabel("Passos de tempo (iterações da simulação)", fontsize=11)
plt.ylabel("Número de nós (indivíduos na rede)", fontsize=11)
plt.title("Dinâmica temporal do modelo SIR competitivo", fontsize=13, pad=12)

plt.legend(
    loc="upper right",
    frameon=True,
    fontsize=10,
    title="Legenda das curvas",
    title_fontsize=10
)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# --- Texto explicativo (para legenda em relatório/slides) ---
print("""
FIGURA 1 — Evolução temporal da disseminação de notícias na rede simulada (N=XXXX).
Curvas sólidas: Infectados ativos por FAKE (vermelho, I_A) e por TRUE (azul, I_B).
Curvas pontilhadas: Recuperados de FAKE (salmão, R_A) e de TRUE (azul-claro, R_B),
indicando nós que já foram atingidos, mas não retransmitem mais.
Linha tracejada cinza: Suscetíveis (S), indivíduos ainda não expostos.
Eixo X: tempo discreto (iterações). Eixo Y: número de nós em cada estado.
""")

# =====================================================
# SEPARAÇÃO DOS NÓS PELO ESTADO FINAL
# =====================================================
final_state = history[-1]  # último passo da simulação

nodes_S  = [n for n, s in final_state.items() if s == 0]
nodes_IA = [n for n, s in final_state.items() if s == 1]
nodes_RA = [n for n, s in final_state.items() if s == 2]
nodes_IB = [n for n, s in final_state.items() if s == 3]
nodes_RB = [n for n, s in final_state.items() if s == 4]

# Subamostragem (para não travar a renderização)
max_draw = 1500
rng = np.random.default_rng(42)
all_nodes = list(G.nodes())
if len(all_nodes) > max_draw:
    nodes_to_draw = rng.choice(all_nodes, size=max_draw, replace=False)
else:
    nodes_to_draw = all_nodes

H = G.subgraph(nodes_to_draw).copy()
pos = nx.spring_layout(H, seed=42, k=0.2)

# =====================================================
# Função auxiliar para plotar uma rede individual
# =====================================================
def plot_subnetwork(H, pos, subset_nodes, color, title, description):
    plt.figure(figsize=(8,6))
    nx.draw_networkx_edges(H, pos, alpha=0.08, width=0.5)
    nx.draw_networkx_nodes(
        H, pos,
        nodelist=[n for n in subset_nodes if n in H.nodes()],
        node_color=color,
        node_size=25,
        alpha=0.9
    )
    plt.title(title, fontsize=13, pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(description)
    print("-"*80, "\n")

# =====================================================
#  1 - Rede dos nós INFECTADOS por FAKE (I_A)
# =====================================================
plot_subnetwork(
    H, pos, nodes_IA, "red",
    "Rede dos nós INFECTADOS por FAKE (I_A)",
    """
FIGURA — Rede dos nós atualmente infectados por conteúdos FAKE (I_A).
Cada nó em vermelho representa um agente que recebeu e está retransmitindo
a informação falsa. As arestas indicam conexões potenciais de disseminação.
A visualização mostra o núcleo ativo da difusão desinformacional no estado final.
    """
)

# =====================================================
#  2 -  Rede dos nós RECUPERADOS de FAKE (R_A)
# =====================================================
plot_subnetwork(
    H, pos, nodes_RA, "salmon",
    "Rede dos nós RECUPERADOS de FAKE (R_A)",
    """
FIGURA — Rede dos nós recuperados de conteúdos FAKE (R_A).
Esses nós foram expostos anteriormente à desinformação, mas já cessaram
sua influência na propagação. Os tons salmão mostram regiões onde houve
maior penetração inicial do conteúdo FAKE durante o processo de difusão.
    """
)

# =====================================================
#  3 - Rede dos nós INFECTADOS por TRUE (I_B)
# =====================================================
plot_subnetwork(
    H, pos, nodes_IB, "blue",
    "Rede dos nós INFECTADOS por TRUE (I_B)",
    """
FIGURA — Rede dos nós atualmente infectados por conteúdos TRUE (I_B).
Os nós azuis indicam agentes ativos na disseminação de informações verídicas.
Esse subgrafo destaca a propagação do conteúdo verdadeiro em competição
com o fluxo de desinformação FAKE.
    """
)

# =====================================================
#  4 -  Rede dos nós RECUPERADOS de TRUE (R_B)
# =====================================================
plot_subnetwork(
    H, pos, nodes_RB, "skyblue",
    "Rede dos nós RECUPERADOS de TRUE (R_B)",
    """
FIGURA — Rede dos nós recuperados de conteúdos TRUE (R_B).
Esses indivíduos já foram alcançados por informações verdadeiras, mas não
estão mais propagando ativamente. Representam o "rastro" de influência
legítima deixado ao longo da simulação.
    """
)

# =====================================================
#  5 -  Rede dos nós SUSCETÍVEIS (S)
# =====================================================
plot_subnetwork(
    H, pos, nodes_S, "lightgray",
    "Rede dos nós SUSCETÍVEIS (S)",
    """
FIGURA — Rede dos nós suscetíveis (S).
Os nós em cinza representam indivíduos que não foram atingidos nem por
conteúdo FAKE nem TRUE até o fim da simulação. Essas regiões da rede
permanecem isoladas ou com baixa conectividade, atuando como bolsões
não expostos à difusão informacional.
    """
)

# =====================================================
# LEGENDA GERAL (com significado dos estados)
# =====================================================
plt.figure(figsize=(7, 1.5))
legend_elements = [
    Patch(facecolor="red", label="I_A — Infectado FAKE"),
    Patch(facecolor="salmon", label="R_A — Recuperado FAKE"),
    Patch(facecolor="blue", label="I_B — Infectado TRUE"),
    Patch(facecolor="skyblue", label="R_B — Recuperado TRUE"),
    Patch(facecolor="lightgray", label="S — Suscetível"),
]
plt.legend(
    handles=legend_elements,
    loc="center",
    frameon=True,
    ncol=3,
    fontsize=9,
    title="Significado das redes individuais",
    title_fontsize=10
)
plt.axis("off")
plt.tight_layout()
plt.show()

# --- velocidade ---
t_peak_A = np.argmax(series["I_A"])
t_peak_B = np.argmax(series["I_B"])
growth_A = series["I_A"][5] - series["I_A"][0]
growth_B = series["I_B"][5] - series["I_B"][0]

# --- alcance ---
total_A = series["I_A"][-1] + series["R_A"][-1]
total_B = series["I_B"][-1] + series["R_B"][-1]
prop_A = total_A / len(G)
prop_B = total_B / len(G)

# --- estrutura ---
nodes_FAKE = [n for n, s in final_state.items() if s in [1, 2]]
nodes_TRUE = [n for n, s in final_state.items() if s in [3, 4]]
G_FAKE = G.subgraph(nodes_FAKE)
G_TRUE = G.subgraph(nodes_TRUE)

diam_FAKE = nx.diameter(G_FAKE) if nx.is_connected(G_FAKE) else np.nan
diam_TRUE = nx.diameter(G_TRUE) if nx.is_connected(G_TRUE) else np.nan

print(f"Tempo de pico — FAKE: {t_peak_A}, TRUE: {t_peak_B}")
print(f"Taxa inicial (ΔI em 5 passos) — FAKE: {growth_A:.1f}, TRUE: {growth_B:.1f}")
print(f"Proporção total atingida — FAKE: {prop_A:.2%}, TRUE: {prop_B:.2%}")
print(f"Diâmetro da maior componente — FAKE: {diam_FAKE}, TRUE: {diam_TRUE}")

# =====================================================
# 1 -  MÁXIMO DE INFECTADOS (pico da difusão)
# =====================================================
t_peak_A = np.argmax(series["I_A"])
t_peak_B = np.argmax(series["I_B"])
max_I_A = int(series["I_A"][t_peak_A])
max_I_B = int(series["I_B"][t_peak_B])

print("\n=== PICO DE INFECTADOS ===")
print(f"- FAKE (I_A): {max_I_A} nós infectados no passo {t_peak_A}")
print(f"- TRUE (I_B): {max_I_B} nós infectados no passo {t_peak_B}")

# --- Figura do pico de infectados ---
plt.figure(figsize=(9,5))
plt.plot(series["I_A"], color="red", linewidth=2.2, label=f"I_A — FAKE (pico = {max_I_A})")
plt.plot(series["I_B"], color="blue", linewidth=2.2, label=f"I_B — TRUE (pico = {max_I_B})")
plt.axvline(x=t_peak_A, color="red", linestyle="--", alpha=0.5)
plt.axvline(x=t_peak_B, color="blue", linestyle="--", alpha=0.5)
plt.xlabel("Passos de tempo")
plt.ylabel("Número de nós infectados")
plt.title("Pico de infectados — SIR competitivo")
plt.legend(frameon=True)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print("""
FIGURA — Curvas de infectados ao longo do tempo.
As linhas verticais tracejadas marcam o passo do pico de infecção:
vermelho = FAKE (I_A) e azul = TRUE (I_B).
O eixo X indica o tempo discreto (iterações do modelo),
e o eixo Y mostra o número de nós simultaneamente infectados.
""")

# =====================================================
# 2 -  CLUSTERING DAS SUB-REDES
# =====================================================

# Define conjuntos de nós relevantes (infectados + recuperados)
final_state = history[-1]
nodes_FAKE = [n for n, s in final_state.items() if s in [1, 2]]  # I_A + R_A
nodes_TRUE = [n for n, s in final_state.items() if s in [3, 4]]  # I_B + R_B

G_FAKE = G.subgraph(nodes_FAKE)
G_TRUE = G.subgraph(nodes_TRUE)

# Calcula clustering médio
clust_FAKE = nx.average_clustering(G_FAKE)
clust_TRUE = nx.average_clustering(G_TRUE)

# Calcula tamanho médio das componentes
components_FAKE = [len(c) for c in nx.connected_components(G_FAKE)] if len(G_FAKE) > 0 else [0]
components_TRUE = [len(c) for c in nx.connected_components(G_TRUE)] if len(G_TRUE) > 0 else [0]

avg_comp_FAKE = np.mean(components_FAKE)
avg_comp_TRUE = np.mean(components_TRUE)

print("\n=== MÉTRICAS DE CLUSTERING E ESTRUTURA ===")
print(f"- Clustering médio FAKE: {clust_FAKE:.4f}")
print(f"- Clustering médio TRUE: {clust_TRUE:.4f}")
print(f"- Tamanho médio das componentes FAKE: {avg_comp_FAKE:.1f} nós")
print(f"- Tamanho médio das componentes TRUE: {avg_comp_TRUE:.1f} nós")

# --- Gráfico comparativo de clustering ---
df_clust = pd.DataFrame({
    "Tipo": ["FAKE", "TRUE"],
    "Clustering médio": [clust_FAKE, clust_TRUE],
    "Tamanho médio das componentes": [avg_comp_FAKE, avg_comp_TRUE]
})

fig, ax1 = plt.subplots(figsize=(7,5))
ax1.bar(df_clust["Tipo"], df_clust["Clustering médio"], color=["red", "blue"], alpha=0.7)
ax1.set_ylabel("Clustering médio", fontsize=11)
ax1.set_title("Comparação do clustering médio entre redes FAKE e TRUE", fontsize=13, pad=10)
ax1.grid(alpha=0.25)
plt.tight_layout()
plt.show()

fig, ax2 = plt.subplots(figsize=(7,5))
ax2.bar(df_clust["Tipo"], df_clust["Tamanho médio das componentes"], color=["salmon", "skyblue"], alpha=0.7)
ax2.set_ylabel("Tamanho médio das componentes", fontsize=11)
ax2.set_title("Tamanho médio das componentes conectadas — FAKE vs TRUE", fontsize=13, pad=10)
ax2.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print("""
FIGURA — Comparação estrutural entre as sub-redes de difusão FAKE e TRUE.
O painel superior mostra o clustering médio (grau de coesão local).
Valores altos indicam que os nós infectados estão próximos de seus vizinhos,
refletindo difusão em comunidades densas.
O painel inferior mostra o tamanho médio das componentes conectadas:
valores maiores indicam difusão mais ampla (atinge mais regiões da rede).
""")
