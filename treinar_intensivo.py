#!/usr/bin/env python3
"""
Script de treinamento otimizado para Q-Learning Hex
Treina automaticamente em fases progressivas para melhor performance
"""

import os
import time
from datetime import datetime
from qlearning_agent import QlearningAgent, comparar_agentes, plotar_metricas

class TreinadorOtimizado:
    def __init__(self):
        self.historico_treinamentos = []
        
    def treinar_fase_progressiva(self):
        """Treina em fases progressivas para otimizar o aprendizado"""
        
        fases = [
            {
                'nome': 'Fundamentos',
                'tamanho': 5,
                'episodios': 50000,
                'oponente': 'aleatorio',
                'alpha': 0.15,
                'gamma': 0.95,
                'epsilon_decay': 0.9995
            },
            {
                'nome': 'Intermediário',
                'tamanho': 7,
                'episodios': 100000,
                'oponente': 'aleatorio',
                'alpha': 0.1,
                'gamma': 0.96,
                'epsilon_decay': 0.9996
            },
            {
                'nome': 'Avançado_Aleatorio',
                'tamanho': 11,
                'episodios': 150000,
                'oponente': 'aleatorio',
                'alpha': 0.08,
                'gamma': 0.97,
                'epsilon_decay': 0.9997
            },
            {
                'nome': 'Elite_vs_Minimax',
                'tamanho': 11,
                'episodios': 200000,
                'oponente': 'minimax',
                'alpha': 0.05,
                'gamma': 0.98,
                'epsilon_decay': 0.9998
            }
        ]
        
        agente_anterior = None
        
        for i, fase in enumerate(fases):
            print(f"\n{'='*60}")
            print(f"🚀 INICIANDO FASE {i+1}: {fase['nome']}")
            print(f"{'='*60}")
            print(f"📋 Configurações:")
            print(f"   - Tabuleiro: {fase['tamanho']}x{fase['tamanho']}")
            print(f"   - Episódios: {fase['episodios']:,}")
            print(f"   - Oponente: {fase['oponente']}")
            print(f"   - α: {fase['alpha']}, γ: {fase['gamma']}")
            print(f"   - Epsilon decay: {fase['epsilon_decay']}")
            
            # Cria novo agente ou transfere conhecimento
            if agente_anterior and fase['tamanho'] == 11:
                print("🔄 Transferindo conhecimento do agente anterior...")
                agente = self.transferir_conhecimento(agente_anterior, fase)
            else:
                agente = QlearningAgent(
                    tamanho_tabuleiro=fase['tamanho'],
                    alpha=fase['alpha'],
                    gamma=fase['gamma'],
                    epsilon=1.0,
                    epsilon_decay=fase['epsilon_decay']
                )
            
            # Nome do arquivo único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            nome_arquivo = f"qlearning_fase{i+1}_{fase['nome']}_{timestamp}.pkl"
            
            # Treina a fase
            inicio = time.time()
            agente.treinar(
                num_episodios=fase['episodios'],
                oponente=fase['oponente'],
                salvar_a_cada=max(1000, fase['episodios']//20),
                nome_arquivo=nome_arquivo
            )
            tempo_fase = time.time() - inicio
            
            # Avalia performance
            self.avaliar_fase(agente, fase, tempo_fase)
            
            # Salva melhor modelo se for a última fase
            if i == len(fases) - 1:
                agente.salvar_modelo("qlearning_MELHOR_MODELO.pkl")
                print("💾 Modelo final salvo como 'qlearning_MELHOR_MODELO.pkl'")
            
            agente_anterior = agente
            
        print("\n🎉 TREINAMENTO PROGRESSIVO CONCLUÍDO!")
        return agente
    
    def transferir_conhecimento(self, agente_origem, config_destino):
        """Transfere conhecimento de um agente menor para um maior"""
        novo_agente = QlearningAgent(
            tamanho_tabuleiro=config_destino['tamanho'],
            alpha=config_destino['alpha'],
            gamma=config_destino['gamma'],
            epsilon=0.5,  # Começa com menos exploração
            epsilon_decay=config_destino['epsilon_decay']
        )
        
        # Copia experiências relevantes (estados menores)
        for (estado, acao), valor in agente_origem.q_table.items():
            tabuleiro_tuple, jogador = estado
            # Se o estado cabe no novo tabuleiro, transfere conhecimento
            if (len(tabuleiro_tuple) <= config_destino['tamanho'] and 
                len(tabuleiro_tuple[0]) <= config_destino['tamanho']):
                novo_agente.q_table[(estado, acao)] = valor * 0.8  # Reduz um pouco a confiança
        
        print(f"📚 Transferidas {len(novo_agente.q_table)} experiências")
        return novo_agente
    
    def avaliar_fase(self, agente, config_fase, tempo_treinamento):
        """Avalia a performance de uma fase de treinamento"""
        print(f"\n📊 AVALIAÇÃO DA FASE {config_fase['nome']}:")
        print(f"⏱️ Tempo de treinamento: {tempo_treinamento/3600:.2f} horas")
        
        total_jogos = agente.vitorias + agente.derrotas + agente.empates
        if total_jogos > 0:
            taxa_vitoria = agente.vitorias / total_jogos * 100
            print(f"🏆 Taxa de vitórias: {taxa_vitoria:.2f}%")
            print(f"📈 Q-table size: {len(agente.q_table):,} entradas")
            print(f"🎯 Epsilon final: {agente.epsilon:.4f}")
            
            # Teste rápido contra minimax se não for a fase de minimax
            if config_fase['oponente'] != 'minimax' and config_fase['tamanho'] >= 7:
                print("🧪 Teste rápido contra Minimax (20 jogos)...")
                v_q, v_m, e = comparar_agentes(agente, 20, config_fase['tamanho'])
                taxa_vs_minimax = v_q / 20 * 100
                print(f"🤖 vs Minimax: {taxa_vs_minimax:.1f}% de vitórias")
    
    def treinar_noturno_automatico(self, duracao_horas=8):
        """Treina automaticamente durante a noite"""
        print(f"🌙 MODO TREINAMENTO NOTURNO - {duracao_horas}h")
        
        inicio = time.time()
        fim_planejado = inicio + (duracao_horas * 3600)
        
        # Estratégia adaptativa baseada no tempo disponível
        if duracao_horas >= 8:
            # Noite completa - treinamento progressivo
            return self.treinar_fase_progressiva()
        else:
            # Tempo limitado - foco em uma fase específica
            return self.treinar_intensivo_rapido(duracao_horas)
    
    def treinar_intensivo_rapido(self, horas_disponiveis):
        """Treinamento intensivo para tempo limitado"""
        episodios = int(horas_disponiveis * 15000)  # ~15k episódios por hora
        
        print(f"⚡ TREINAMENTO INTENSIVO - {episodios:,} episódios")
        
        agente = QlearningAgent(
            tamanho_tabuleiro=11,
            alpha=0.1,
            gamma=0.97,
            epsilon=1.0,
            epsilon_decay=0.9995
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        nome_arquivo = f"qlearning_intensivo_{episodios//1000}k_{timestamp}.pkl"
        
        agente.treinar(
            num_episodios=episodios,
            oponente='aleatorio',
            salvar_a_cada=max(500, episodios//20),
            nome_arquivo=nome_arquivo
        )
        
        return agente

def menu_treinamento_otimizado():
    """Menu para escolher o tipo de treinamento"""
    print("🎯 TREINAMENTO OTIMIZADO Q-LEARNING")
    print("="*50)
    print("1. 🌙 Treinamento Noturno Automático (8h)")
    print("2. ⚡ Treinamento Intensivo Rápido")
    print("3. 🚀 Treinamento Progressivo Completo")
    print("4. 🧪 Teste de Configuração Personalizada")
    print("5. 🔄 Continuar treinamento existente")
    
    escolha = input("Escolha uma opção: ")
    
    treinador = TreinadorOtimizado()
    
    if escolha == "1":
        horas = int(input("Quantas horas disponíveis (padrão 8)? ") or "8")
        agente = treinador.treinar_noturno_automatico(horas)
        
    elif escolha == "2":
        horas = int(input("Quantas horas disponíveis? "))
        agente = treinador.treinar_intensivo_rapido(horas)
        
    elif escolha == "3":
        print("🚀 Iniciando treinamento progressivo completo...")
        print("⚠️  Isso pode levar 20-30 horas total!")
        confirmar = input("Continuar? (s/n): ").lower()
        if confirmar == 's':
            agente = treinador.treinar_fase_progressiva()
        else:
            return
            
    elif escolha == "4":
        agente = treinar_personalizado()
        
    elif escolha == "5":
        agente = continuar_treinamento_existente()
    
    else:
        print("Opção inválida!")
        return
    
    # Opções pós-treinamento
    print("\n🎯 TREINAMENTO CONCLUÍDO!")
    print("O que deseja fazer?")
    print("1. 🤖 Testar contra Minimax")
    print("2. 🎮 Jogar contra o agente")
    print("3. 📊 Ver métricas de treinamento")
    
    pos_opcao = input("Escolha: ")
    
    if pos_opcao == "1":
        num_jogos = int(input("Número de jogos (padrão 100): ") or "100")
        comparar_agentes(agente, num_jogos, agente.tamanho_tabuleiro)
    elif pos_opcao == "2":
        agente.jogar_contra_humano()
    elif pos_opcao == "3":
        plotar_metricas(agente)

def treinar_personalizado():
    """Permite configuração personalizada de treinamento"""
    print("\n🔧 CONFIGURAÇÃO PERSONALIZADA")
    
    tamanho = int(input("Tamanho do tabuleiro (5-11): ") or "11")
    episodios = int(input("Número de episódios: ") or "100000")
    
    print("Oponente: 1-Aleatório, 2-Minimax")
    oponente = "minimax" if input("Escolha: ") == "2" else "aleatorio"
    
    alpha = float(input("Taxa de aprendizado α (0.01-0.2): ") or "0.1")
    gamma = float(input("Fator de desconto γ (0.9-0.99): ") or "0.95")
    
    agente = QlearningAgent(
        tamanho_tabuleiro=tamanho,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=0.9995
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    nome_arquivo = f"qlearning_custom_{timestamp}.pkl"
    
    agente.treinar(
        num_episodios=episodios,
        oponente=oponente,
        salvar_a_cada=max(500, episodios//20),
        nome_arquivo=nome_arquivo
    )
    
    return agente

def continuar_treinamento_existente():
    """Continua o treinamento de um modelo existente"""
    print("\n🔄 CONTINUAR TREINAMENTO")
    
    modelos = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if not modelos:
        print("Nenhum modelo encontrado!")
        return None
    
    print("Modelos disponíveis:")
    for i, modelo in enumerate(modelos, 1):
        print(f"{i}. {modelo}")
    
    try:
        escolha = int(input("Escolha um modelo: ")) - 1
        nome_arquivo = modelos[escolha]
    except:
        print("Escolha inválida!")
        return None
    
    agente = QlearningAgent()
    agente.carregar_modelo(nome_arquivo)
    
    episodios_extras = int(input("Quantos episódios adicionais: ") or "50000")
    
    print("Oponente: 1-Aleatório, 2-Minimax")
    oponente = "minimax" if input("Escolha: ") == "2" else "aleatorio"
    
    # Reduz epsilon para menos exploração (já aprendeu bastante)
    agente.epsilon = max(agente.epsilon, 0.1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    novo_nome = f"{nome_arquivo.split('.')[0]}_cont_{timestamp}.pkl"
    
    agente.treinar(
        num_episodios=episodios_extras,
        oponente=oponente,
        salvar_a_cada=max(500, episodios_extras//20),
        nome_arquivo=novo_nome
    )
    
    return agente

if __name__ == "__main__":
    menu_treinamento_otimizado()