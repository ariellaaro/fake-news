''' ================================
additional libraries
================================ '''
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import community.community_louvain as community_louvain

''' ================================
dataset path (change as needed)
================================ '''

# the following runs on Google Colab; as an alternative, you may download the dataset via git LFS from the repo
from google.colab import drive
drive.mount('/content/drive')

csv_path = "/content/drive/MyDrive/IME/MC_Fake_dataset.csv"
df = pd.read_csv(csv_path)

''' ================================
subset selection (optional)
================================ '''

'''
# subset - all
subset = ["gossipcop", "RealPolitics", "FA-KES", "politifact", "RealSyria", "HealthStory", "HealthRelease", "FakeCovid", "FakeCovidClaimFiltered", "RealCovid", "RealHealth"]

# subset - health
subset = ["HealthStory", "HealthRelease", "FakeCovid", "FakeCovidClaimFiltered", "RealCovid", "RealHealth"]

# subset - politics
subset = ["RealPolitics", "FA-KES", "politifact", "RealSyria"]

# subset - balanced
subset = ["FA-KES", "politifact", "HealthStory", "HealthRelease", "FakeCovid", "FakeCovidClaimFiltered"]
'''

df = df[df["data_name"].isin(subset)].copy()

print("News per category:")
for name, count in df["data_name"].value_counts().items():
    print(f"  {name}: {count}")
print()

print("Labels per category (0/1):")
label_counts = df.groupby("data_name")["labels"].value_counts().unstack(fill_value=0)
for name, row in label_counts.iterrows():
    print(f"  {name}: {row[0]}, {row[1]}")

total_0 = label_counts[0].sum()
total_1 = label_counts[1].sum()

print("\nTotal:")
print(f"  Label 0 (TRUE): {label_counts[0].sum()}")
print(f"  Label 1 (FAKE): {total_1}")

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


def diffusion_metrics(G: nx.DiGraph):
    if G.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0, "avg_degree": 0, "density": 0}
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "density": nx.density(G),
    }

''' ================================
network metrics
================================ '''

G_fake = build_graph(df[df["labels"] == "fake"])
G_true = build_graph(df[df["labels"] == "true"])

fake_stats = diffusion_metrics(G_fake)
true_stats = diffusion_metrics(G_true)

print("Fake news:", fake_stats)
print("True news:", true_stats)

''' ================================
graph visualization
================================ '''

def draw_graph(G, title="", max_nodes=500, layout="spring", node_color="blue"):

    if G.number_of_nodes() == 0:
        print(f"[{title}] Empty graph.")
        return

    if G.number_of_nodes() > max_nodes:
        nodes_sample = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes_sample).copy()
        print(f"[{title}] showing only {len(nodes_sample)} nodes from {G.number_of_nodes()}")

    plt.figure(figsize=(8, 6), dpi=300)

    if layout == "spring":
        pos = nx.spring_layout(G, k=0.2, iterations=30)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.random_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=40, alpha=0.7, node_color=node_color)
    nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.4, width=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()

draw_graph(G_fake, title="Fake news network (from sample)", max_nodes=500, node_color="brown")
draw_graph(G_true, title="True news network (from sample)", max_nodes=500, node_color="royalblue")

''' ================================
community metrics - clustering coefficient
================================ '''

def calculate_clustering(G, title=""):

    if G.number_of_nodes() == 0:
        print(f"[{title}] Empty graph.")
        return

    G_undirected = G.to_undirected()

    avg_clustering = nx.average_clustering(G_undirected)

    print(f"\n{title}")
    print(f"  Average clustering coefficient: {avg_clustering:.4f}")

    return avg_clustering

clustering_fake = calculate_clustering(G_fake, title="FAKE NEWS")
clustering_true = calculate_clustering(G_true, title="TRUE NEWS")

''' ================================
community metrics - louvain community detection
================================ '''

def detect_communities(G, title=""):

    if G.number_of_nodes() == 0:
        print(f"[{title}] Empty graph.")
        return None, 0, 0

    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected)
    modularity = community_louvain.modularity(partition, G_undirected)

    num_communities = len(set(partition.values()))

    print(f"\n{title}")
    print(f"  Number of communities: {num_communities}")
    print(f"  Modularity: {modularity:.4f}")

    return partition, modularity, num_communities

partition_fake, mod_fake, num_comm_fake = detect_communities(G_fake, title="FAKE NEWS")
partition_true, mod_true, num_comm_true = detect_communities(G_true, title="TRUE NEWS")
