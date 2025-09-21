import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def gera_grafo_erdos_renyi(n, p):
    return nx.erdos_renyi_graph(n, p)

def analisa_grafo(G):
    largest_cc = max(nx.connected_components(G), key=len)
    gcc_size = len(largest_cc)
    total_non_gcc_size = 0
    num_non_gcc_components = 0
    for cc in tqdm.tqdm(nx.connected_components(G), desc="Analisando componentes", leave=False):
        if cc != largest_cc:
            total_non_gcc_size += len(cc)
            num_non_gcc_components += 1
    avg_isolated_size = (total_non_gcc_size / num_non_gcc_components) if num_non_gcc_components > 0 else 0
    return gcc_size, avg_isolated_size

def realiza_experimento(n, num_instances, k_values):
    avg_gcc_sizes = []
    avg_isolated_components_sizes = []
    for k in tqdm.tqdm(k_values, desc="Realizando Experimento"):
        p = k / (n - 1)
        gcc_sizes = []
        isolated_components_sizes_per_k = []
        for _ in range(num_instances):
            G = gera_grafo_erdos_renyi(n, p)
            gcc_size, avg_isolated_size = analisa_grafo(G)
            gcc_sizes.append(gcc_size / n)
            isolated_components_sizes_per_k.append(avg_isolated_size)
        avg_gcc_sizes.append(np.mean(gcc_sizes))
        avg_isolated_components_sizes.append(np.mean(isolated_components_sizes_per_k))
    return avg_gcc_sizes, avg_isolated_components_sizes

def plota_gcc_k_medio(k_values, avg_gcc_sizes):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_gcc_sizes, label='Tamanho da GCC / N')
    plt.axvline(x=1, color='r', linestyle='--', label='Ponto Crítico (<k>=1)')
    plt.title('Tamanho Médio da Maior Componente Conexa (GCC) vs. Grau Médio')
    plt.xlabel('Grau Médio (<k>)')
    plt.ylabel('Tamanho Médio da GCC / N')
    plt.grid(True)
    plt.legend()
    plt.savefig('gcc_plot.png')
    plt.show()

def plota_gcc_k_medio_log(k_values, avg_gcc_sizes):
    plt.figure(figsize=(10, 6))
    plt.semilogx(k_values, avg_gcc_sizes)
    plt.axvline(x=1, color='r', linestyle='--')
    plt.title('GCC vs. Grau Médio (eixo x em escala log)')
    plt.xlabel('Grau Médio (<k>, escala log)')
    plt.ylabel('Tamanho Médio da GCC / N')
    plt.grid(True)
    plt.savefig('gcc_log_plot.png')
    plt.show()

def plota_isolados_k(k_values, avg_isolated_components_sizes):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, avg_isolated_components_sizes, color='g', label='Tamanho Médio das Componentes Isoladas')
    plt.axvline(x=1, color='r', linestyle='--', label='Ponto Crítico (<k>=1)')
    plt.title('Tamanho Médio das Componentes Isoladas vs. Grau Médio')
    plt.xlabel('Grau Médio (<k>)')
    plt.ylabel('Tamanho Médio das Componentes Isoladas')
    plt.grid(True)
    plt.legend()
    plt.savefig('isolated_components_plot.png')
    plt.show()

def main():
    n = 100
    num_instances = 500
    k_values = np.linspace(0, 6, 100)
    avg_gcc_sizes, avg_isolated_components_sizes = realiza_experimento(n, num_instances, k_values)
    plota_gcc_k_medio(k_values, avg_gcc_sizes)
    plota_gcc_k_medio_log(k_values, avg_gcc_sizes)
    plota_isolados_k(k_values, avg_isolated_components_sizes)

if __name__ == "__main__":
    main()
