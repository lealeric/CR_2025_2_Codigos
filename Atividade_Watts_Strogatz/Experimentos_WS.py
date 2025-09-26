import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


def _executar_experimento_probabilidade(args):
    """
    Função auxiliar para execução paralela de experimentos.

    Args:
        args: tupla contendo (p, n, k, num_experimentos, distancia_media0, cc0, cc_const)

    Returns:
        dict com resultados dos experimentos para a probabilidade p
    """
    p, n, k, num_experimentos, distancia_media0, cc0, cc_const = args

    valores_experimento = {
        'proporcao_Cp_C0': [],
        'proporcao_Lp_L0': [],
        'porporcao_CC_G_C0_expected': [],
    }

    for _ in range(num_experimentos):
        g = nx.watts_strogatz_graph(n, k, p)

        dist_media = nx.average_shortest_path_length(g)
        valores_experimento['proporcao_Lp_L0'].append(dist_media / distancia_media0)

        cc_g = nx.cluster.average_clustering(g)
        valores_experimento['proporcao_Cp_C0'].append(cc_g / cc0)

        cc_esperado = cc_const * ((1 - p) ** 3)
        if cc_esperado == 0:
            valores_experimento['porporcao_CC_G_C0_expected'].append(0)
        else:
            valores_experimento['porporcao_CC_G_C0_expected'].append(cc_g / cc_esperado)

    return {
        'p': p,
        'proporcao_Cp_C0': np.mean(valores_experimento['proporcao_Cp_C0']),
        'proporcao_Lp_L0': np.mean(valores_experimento['proporcao_Lp_L0']),
        'porporcao_CC_G_C0_expected': np.mean(valores_experimento['porporcao_CC_G_C0_expected'])
    }


class ExperimentoWS:
    def __init__(self, n, k, probabilidades, num_experimentos=20, num_processos=None):
        self.n = n
        self.k = k
        self.probabilidades = probabilidades
        self.num_experimentos = num_experimentos
        self.num_processos = num_processos or mp.cpu_count()
        
        self.G0 = nx.watts_strogatz_graph(n, k, 0)
        self.distancia_media0 = nx.average_shortest_path_length(self.G0)
        self.CC0 = nx.cluster.average_clustering(self.G0)

        self.cc_const = (3 * (k - 1)) / (2 * (2 * k - 1))

        self.valores_teste = {
            'proporcao_Cp_C0': [],
            'proporcao_Lp_L0': [],
            'valor_p': [],
            'porporcao_CC_G_C0_expected': [],
        }

    def return_proporcao_distancia(self, G):
        """Calcula proporção da distância média com otimização.

        Args:
            G (networkx.Graph): Grafo para o qual calcular a proporção da distância média.

        Returns:
            float: Proporção da distância média do grafo G em relação à distância média do grafo inicial G0.
        """
        return nx.average_shortest_path_length(G) / self.distancia_media0

    def return_proporcao_CC(self, G):
        """Calcula proporção do coeficiente de clustering com otimização.

        Args:
            G (networkx.Graph): Grafo para o qual calcular a proporção do coeficiente de clustering.

        Returns:
            float: Proporção do coeficiente de clustering do grafo G em relação ao coeficiente de clustering do grafo inicial G0.
        """
        return nx.cluster.average_clustering(G) / self.CC0

    def return_proporcao_CC_expected(self, G, p):
        """Calcula proporção CC/CC_esperado com otimização.

        Args:
            G (networkx.Graph): Grafo para o qual calcular a proporção CC/CC_esperado.
            p (float): Probabilidade de rewire.

        Returns:
            float: Proporção CC/CC_esperado do grafo G em relação ao grafo inicial G0.
        """
        CC_esperado = self.cc_const * ((1 - p) ** 3)
        CC_G = nx.cluster.average_clustering(G)
        if CC_esperado == 0:
            return 0
        return CC_G / CC_esperado

    def run_experiments(self):
        """Executa experimentos para todas as probabilidades em paralelo."""
        args_list = [
            (p, self.n, self.k, self.num_experimentos, 
             self.distancia_media0, self.CC0, self.cc_const)
            for p in self.probabilidades
        ]

        with mp.Pool(processes=self.num_processos) as pool:
            resultados = list(tqdm(
                pool.imap_unordered(_executar_experimento_probabilidade, args_list),
                total=len(args_list),
                desc="Executando experimentos"
            ))
            
        resultados.sort(key=lambda x: x['p'])

        for resultado in resultados:
            self.valores_teste['valor_p'].append(resultado['p'])
            self.valores_teste['proporcao_Cp_C0'].append(resultado['proporcao_Cp_C0'])
            self.valores_teste['proporcao_Lp_L0'].append(resultado['proporcao_Lp_L0'])
            self.valores_teste['porporcao_CC_G_C0_expected'].append(
                resultado['porporcao_CC_G_C0_expected'])

    def plot_results(self):
        """Plota os resultados dos experimentos."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.valores_teste['valor_p'],
                 self.valores_teste['proporcao_Cp_C0'],
                 label='Cp/C0', color='blue')
        plt.plot(self.valores_teste['valor_p'],
                 self.valores_teste['proporcao_Lp_L0'],
                 label='Lp/L0', color='orange')
        plt.plot(self.valores_teste['valor_p'],
                 self.valores_teste['porporcao_CC_G_C0_expected'],
                 label='CC_G/CC_esperado', color='green')

        plt.xscale('log')
        plt.ylim(0, 1.05)
        plt.xlabel("p")
        plt.ylabel("Proporções normalizadas")
        plt.legend()
        plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
        plt.savefig("resultados_experimento_ws.png")
        plt.show()


def main():
    """Função principal para executar os experimentos Watts-Strogatz."""
    n = 1000
    k = 10
    probabilidades = np.linspace(0.0001, 1, 5000)

    experimento = ExperimentoWS(n, k, probabilidades)
    experimento.run_experiments()
    experimento.plot_results()


if __name__ == "__main__":
    mp.freeze_support()
    main()
