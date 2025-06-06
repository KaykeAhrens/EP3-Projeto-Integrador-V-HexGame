#!/usr/bin/env python3
"""
Script de Treinamento Intensivo para Agente Hex Q-Learning 11x11
Versão otimizada para tabuleiros maiores
"""

import os
import time
from datetime import datetime
from qlearning_agent import QlearningAgent, comparar_agentes, plotar_metricas

def treinar_agente_11x11():
    """Treina o agente especificamente para tabuleiro 11x11"""
    
    print("🚀 TREINAMENTO INTENSIVO HEX 11x11 INICIADO")
    print("=" * 60)
    
    # Parâmetros OTIMIZADOS para 11x11
    PARAMETROS = {
        'tamanho_tabuleiro': 11,  # Mudança principal
        'alpha': 0.1,            # Menor para mais estabilidade
        'gamma': 0.98,           # Maior para valorizar jogadas futuras
        'epsilon': 1.0,
        'epsilon_decay': 0.99985,  # Decay mais lento
        'epsilon_min': 0.02      # Exploração mínima menor
    }
    
    # Verifica modelo existente
    modelo_existente = None
    for arquivo in os.listdir('.'):
        if arquivo.startswith('hex_11x11_') and arquivo.endswith('.pkl'):
            modelo_existente = arquivo
            break
    
    if modelo_existente:
        print(f"📂 Modelo 11x11 existente encontrado: {modelo_existente}")
        continuar = input("Deseja continuar o treinamento existente? (s/n): ").lower()
        if continuar == 's':
            agente = QlearningAgent(**PARAMETROS)
            agente.carregar_modelo(modelo_existente)
            print("✅ Modelo carregado com sucesso!")
        else:
            agente = QlearningAgent(**PARAMETROS)
    else:
        agente = QlearningAgent(**PARAMETROS)
    
    # === FASE 1: FUNDAMENTOS ESTENDIDOS ===
    print("\n🎯 FASE 1: FUNDAMENTOS BÁSICOS 11x11 (vs Aleatório)")
    print("Objetivo: Mapear estados básicos do tabuleiro maior")
    
    fase1_nome = f"hex_11x11_fase1_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    
    try:
        # MAIS episódios para tabuleiro maior
        agente.treinar(
            num_episodios=75000,  # 3x mais que 7x7
            oponente='aleatorio',
            salvar_a_cada=5000,
            nome_arquivo=fase1_nome
        )
        
        print("\n📊 AVALIAÇÃO FASE 1:")
        avaliar_agente_11x11(agente, "Fase 1")
        
    except KeyboardInterrupt:
        print("\n⏸️ Treinamento interrompido pelo usuário")
        agente.salvar_modelo(fase1_nome)
        return agente
    
    # === FASE 2: DESENVOLVIMENTO TÁTICO ===
    print("\n🧠 FASE 2: DESENVOLVIMENTO TÁTICO (vs Minimax Leve)")
    print("Objetivo: Aprender padrões táticos específicos do 11x11")
    
    # Ajustes para fase intermediária
    agente.epsilon = 0.4  # Mais exploração que no 7x7
    agente.alpha = 0.08
    
    fase2_nome = f"hex_11x11_fase2_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    
    try:
        agente.treinar(
            num_episodios=50000,  # Mais episódios
            oponente='minimax',
            salvar_a_cada=5000,
            nome_arquivo=fase2_nome
        )
        
        print("\n📊 AVALIAÇÃO FASE 2:")
        avaliar_agente_11x11(agente, "Fase 2")
        
    except KeyboardInterrupt:
        print("\n⏸️ Treinamento interrompido pelo usuário")
        agente.salvar_modelo(fase2_nome)
        return agente
    
    # === FASE 3: ESTRATÉGIA AVANÇADA ===
    print("\n⚔️ FASE 3: ESTRATÉGIA AVANÇADA 11x11")
    print("Objetivo: Dominar estratégias de longo prazo")
    
    agente.epsilon = 0.15  # Ainda alguma exploração
    agente.alpha = 0.05
    
    fase3_nome = f"hex_11x11_fase3_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    
    try:
        agente.treinar(
            num_episodios=40000,
            oponente='minimax',
            salvar_a_cada=4000,
            nome_arquivo=fase3_nome
        )
        
        print("\n📊 AVALIAÇÃO FASE 3:")
        avaliar_agente_11x11(agente, "Fase 3")
        
    except KeyboardInterrupt:
        print("\n⏸️ Treinamento interrompido pelo usuário")
        agente.salvar_modelo(fase3_nome)
        return agente
    
    # === FASE 4: POLIMENTO FINAL ===
    print("\n✨ FASE 4: POLIMENTO FINAL 11x11")
    print("Objetivo: Refinamento e consistência")
    
    agente.epsilon = 0.05
    agente.alpha = 0.03
    
    fase4_nome = f"hex_11x11_FINAL_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    
    try:
        agente.treinar(
            num_episodios=25000,
            oponente='minimax',
            salvar_a_cada=2500,
            nome_arquivo=fase4_nome
        )
        
        print("\n🏆 AVALIAÇÃO FINAL:")
        avaliar_agente_11x11(agente, "Final")
        
    except KeyboardInterrupt:
        print("\n⏸️ Treinamento interrompido pelo usuário")
        agente.salvar_modelo(fase4_nome)
        return agente
    
    # Salva modelo final
    modelo_final = f"hex_11x11_PRONTO_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    agente.salvar_modelo(modelo_final)
    
    print(f"\n🎉 TREINAMENTO 11x11 COMPLETO!")
    print(f"📁 Modelo final: {modelo_final}")
    print(f"📊 Total de episódios: ~190.000")
    print("🎮 Agente pronto para Hex 11x11!")
    
    return agente

def avaliar_agente_11x11(agente, fase):
    """Avaliação específica para 11x11"""
    total_jogos = agente.vitorias + agente.derrotas + agente.empates
    
    if total_jogos > 0:
        taxa_vitoria = agente.vitorias / total_jogos * 100
        print(f"   📈 Taxa de vitórias: {taxa_vitoria:.1f}%")
        print(f"   🎯 Total de jogos: {total_jogos}")
        print(f"   🧠 Estados mapeados: {len(agente.q_table)}")
        print(f"   🔍 Epsilon atual: {agente.epsilon:.4f}")
        
        # Estatísticas específicas para 11x11
        media_jogadas = total_jogos // max(1, len(agente.q_table) // 1000)
        print(f"   ⏱️ Média estados/jogo: ~{media_jogadas}")
    
    # Teste mais rigoroso para 11x11
    print(f"   ⚔️ Teste {fase} vs Minimax (30 jogos - 11x11)...")
    vit_q, vit_m, emp = comparar_agentes(agente, num_jogos=30, tamanho_tabuleiro=11)
    taxa_contra_minimax = vit_q / 30 * 100
    print(f"   🏆 Performance vs Minimax: {taxa_contra_minimax:.1f}% vitórias")
    
    if taxa_contra_minimax > 40:
        print(f"   🌟 Excelente! Para 11x11, isso é muito bom!")
    elif taxa_contra_minimax > 25:
        print(f"   👍 Bom progresso para tabuleiro 11x11")
    else:
        print(f"   📚 Ainda aprendendo... 11x11 é desafiador!")

def treinar_progressivo_11x11():
    """Treinamento progressivo: 7x7 -> 9x9 -> 11x11"""
    print("🎯 TREINAMENTO PROGRESSIVO PARA 11x11")
    print("Estratégia: Começar pequeno e expandir gradualmente")
    
    # Fase 1: Treina em 7x7
    print("\n📚 ETAPA 1: Fundamentos em 7x7")
    agente_base = QlearningAgent(
        tamanho_tabuleiro=7,
        alpha=0.15,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05
    )
    
    agente_base.treinar(
        num_episodios=20000,
        oponente='aleatorio',
        salvar_a_cada=5000,
        nome_arquivo='progresso_7x7.pkl'
    )
    
    # Fase 2: Expande para 9x9
    print("\n🔄 ETAPA 2: Expansão para 9x9")
    agente_medio = QlearningAgent(
        tamanho_tabuleiro=9,
        alpha=0.12,
        gamma=0.96,
        epsilon=0.5,  # Já tem algum conhecimento
        epsilon_decay=0.9998,
        epsilon_min=0.03
    )
    
    agente_medio.treinar(
        num_episodios=30000,
        oponente='aleatorio',
        salvar_a_cada=5000,
        nome_arquivo='progresso_9x9.pkl'
    )
    
    # Fase 3: Final em 11x11
    print("\n🚀 ETAPA 3: Domínio do 11x11")
    agente_final = QlearningAgent(
        tamanho_tabuleiro=11,
        alpha=0.1,
        gamma=0.98,
        epsilon=0.3,  # Conhecimento prévio
        epsilon_decay=0.99985,
        epsilon_min=0.02
    )
    
    agente_final.treinar(
        num_episodios=60000,
        oponente='minimax',
        salvar_a_cada=6000,
        nome_arquivo='hex_11x11_progressivo_FINAL.pkl'
    )
    
    print("\n✅ TREINAMENTO PROGRESSIVO CONCLUÍDO!")
    return agente_final

if __name__ == "__main__":
    print("🎯 SISTEMA DE TREINAMENTO HEX 11x11")
    print("=" * 50)
    print("1. 🚀 Treinamento Intensivo Direto 11x11 (~190k episódios)")
    print("2. 📈 Treinamento Progressivo 7x7→9x9→11x11 (~110k episódios)")
    print("3. 🔄 Continuar Modelo 11x11 Existente")
    print("4. 🎮 Testar Modelo 11x11")
    
    opcao = input("\nEscolha uma opção: ").strip()
    
    if opcao == "1":
        print("\n⚠️  AVISO: Treinamento intensivo pode levar várias horas!")
        confirmar = input("Continuar? (s/n): ").lower()
        if confirmar == 's':
            agente = treinar_agente_11x11()
            
            jogar = input("\n🎮 Jogar contra o agente 11x11? (s/n): ").lower()
            if jogar == 's':
                agente.jogar_contra_humano()
                
    elif opcao == "2":
        print("\n📚 Estratégia mais eficiente - recomendada!")
        agente = treinar_progressivo_11x11()
        
        jogar = input("\n🎮 Jogar contra o agente? (s/n): ").lower()
        if jogar == 's':
            agente.jogar_contra_humano()
            
    elif opcao == "3":
        # Lista apenas modelos 11x11
        modelos = [f for f in os.listdir('.') if f.endswith('.pkl') and '11x11' in f]
        
        if not modelos:
            print("❌ Nenhum modelo 11x11 encontrado!")
        else:
            print("📁 Modelos 11x11 disponíveis:")
            for i, modelo in enumerate(modelos, 1):
                print(f"{i}. {modelo}")
            
            try:
                escolha = int(input("Escolha: ")) - 1
                modelo_escolhido = modelos[escolha]
                
                agente = QlearningAgent(tamanho_tabuleiro=11)
                agente.carregar_modelo(modelo_escolhido)
                
                episodios = int(input("Episódios extras (padrão 20000): ") or "20000")
                agente.treinar(
                    num_episodios=episodios,
                    oponente='minimax',
                    salvar_a_cada=max(2000, episodios//10),
                    nome_arquivo=modelo_escolhido.replace('.pkl', f'_plus{episodios}.pkl')
                )
                
            except:
                print("❌ Erro!")
                
    elif opcao == "4":
        modelos = [f for f in os.listdir('.') if f.endswith('.pkl') and '11x11' in f]
        
        if not modelos:
            print("❌ Nenhum modelo 11x11 encontrado!")
        else:
            print("📁 Modelos 11x11:")
            for i, modelo in enumerate(modelos, 1):
                print(f"{i}. {modelo}")
            
            try:
                escolha = int(input("Escolha: ")) - 1
                modelo_escolhido = modelos[escolha]
                
                agente = QlearningAgent(tamanho_tabuleiro=11)
                agente.carregar_modelo(modelo_escolhido)
                
                avaliar_agente_11x11(agente, "Teste")
                
                jogar = input("\n🎮 Jogar? (s/n): ").lower()
                if jogar == 's':
                    agente.jogar_contra_humano()
                    
            except:
                print("❌ Erro ao carregar!")
    
    else:
        print("❌ Opção inválida!")
    
    print("\n👋 Finalizado!")