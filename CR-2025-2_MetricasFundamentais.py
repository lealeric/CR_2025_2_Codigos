import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_degree_distribution_complementar(grafos, titles=None, output_path: str = 'E://', output_file_name: str = 'degree_distribution.png', show: bool = False):
    """Plota o gráfico de dispersão da distribuição complementar do grau.
    Cria um layout 2x2: cada grafo é plotado em escala linear (esquerda) e log-log (direita).
    Primeiro grafo na parte superior, segundo grafo na parte inferior.

    Args:
        grafos (list or nx.Graph): Lista de grafos do NetworkX ou um único grafo.
        titles (list): Lista de títulos para cada grafo.
        output_path (str): Caminho de saída do arquivo.
        output_file_name (str): Nome do arquivo de saída.
        show (bool): Se o gráfico deve ser exibido.                
    """
    # Se apenas um grafo for passado, converte para lista
    if not isinstance(grafos, list):
        grafos = [grafos]
        
    if titles is None:
        titles = [f"Grafo {i+1}" for i in range(len(grafos))]
    
    # Determina o número de linhas baseado no número de grafos
    num_grafos = len(grafos)
    
    # Cria subplots em layout 2x2 (ou 1x2 se apenas um grafo)
    _, axes = plt.subplots(num_grafos, 2, figsize=(16, 8 * num_grafos))
    
    # Se apenas um grafo, axes precisa ser reformatado para 2D
    if num_grafos == 1:
        axes = axes.reshape(1, -1)
    
    for i, (grafo, title) in enumerate(zip(grafos, titles)):
        degrees = [degree for node, degree in grafo.degree()]
        degrees.sort()
        prob = [1 - (j/len(degrees)) for j in range(len(degrees))]

        # Gráfico linear (coluna da esquerda)
        axes[i, 0].scatter(degrees, prob, alpha=0.7)
        axes[i, 0].set_xlabel("Grau")
        axes[i, 0].set_ylabel("Probabilidade Complementar")
        axes[i, 0].set_title(f"{title} - Escala Linear")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylim(0, 1)
        
        # Gráfico log-log (coluna da direita)        
        axes[i, 1].scatter(degrees, prob, alpha=0.7)
        axes[i, 1].set_xscale('log')
        axes[i, 1].set_yscale('log')
        axes[i, 1].set_xlabel("Grau (log)")
        axes[i, 1].set_ylabel("Probabilidade Complementar (log)")
        axes[i, 1].set_title(f"{title} - Escala Log-Log")
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_path}{output_file_name}", dpi=300, bbox_inches='tight')
    
    print(f"Gráfico salvo em: {output_path}{output_file_name}")
    
    if show:
        plt.show()

    plt.close()
    
def show_graph_metrics(graph: nx.Graph, *, clusters: bool = False) -> None:
    """
    Show some metrics of the graph.
    
    Args:
        graph (nx.graph): A graph object.
        clusters (bool, optional): If the clusters should be displayed. Defaults to False.
        
    Example:
        >>> G = nx.Graph()
        >>> G.add_node(1)
        >>> G.add_node(2)
        >>> G.add_edge(1, 2)
        >>> show_graph_metrics(G)
        Nº de nós: 2
        Nº de arestas: 1
        Grau médio: 1.0
        Densidade: 1.0
    """    
    degrees = []

    for node in graph.nodes():
        degrees.append(nx.degree(graph, node))

    print(f"Nº de nós: {graph.number_of_nodes():,}")
    print(f"Nº de arestas: {graph.number_of_edges():,}")
    print(f"Densidade: {nx.density(graph):.6f}")
    print(f"Grau médio: {np.mean(degrees):.6f}")
    print(f"Centralidade de grau médio: {np.mean(list(nx.degree_centrality(graph).values())):.6f}")
    # print(f"Diâmetro: {nx.diameter(graph):,.0f}")


    if clusters:
        print(f"Cluster global: {nx.transitivity(graph):.6f}")
        print(f"Cluster médio: {nx.average_clustering(graph):.6f}")

def main():
    grafo_atores = nx.read_edgelist('actor.edgelist.txt')
    grafo_web = nx.read_edgelist('www.edgelist.txt')

    print("Métricas do grafo de atores:")
    show_graph_metrics(grafo_atores, clusters=True)
    
    print("-"*30)

    print("Métricas do grafo da web:")
    show_graph_metrics(grafo_web, clusters=True)

    plot_degree_distribution_complementar(
        [grafo_atores, grafo_web],
        ["Grafo de Atores", "Grafo da Web"]
    )


if __name__ == "__main__":
    main()