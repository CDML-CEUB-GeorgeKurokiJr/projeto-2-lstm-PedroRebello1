# Aviso Importante

É fundamental destacar logo de início que embora o resultado desta versão pareça visualmente perfeito nos gráficos, o modelo está viciado. 

A rede neural aprendeu uma heurística simples: ela está apenas copiando o valor de fechamento do dia anterior e utilizando-o como a previsão do dia seguinte.

Diversos passos e abordagens deram errado no processo de modelagem, incluindo:
 - Testes com múltiplos otimizadores.
 - Uso de diferentes funções de loss.
 - Mudança da variável alvo, tentando prever a flutuação/delta em vez do valor absoluto.


Em outras iterações, o modelo acabava penalizado e passava a prever apenas a média histórica dos preços, criando uma reta inútil.

Diante disso, foi preferível manter este modelo viciado, pois ele ilustra o funcionamento do pipeline, mesmo sendo não funcional para qualquer operação real ou previsão de mercado.

---

# Comportamento dos Ativos 
Ao observar os dados e os resultados, fica claro que o valor da maioria das ações escolhidas (NVDA, AMD, TSM, WDC) segue um padrão direcional muito parecido. A única exceção notória é a Intel (INTC).

## Por que isso acontece?
A correlação entre NVDA, AMD, TSM e WDC reflete o ecossistema atual de semicondutores e o boom da Inteligência Artificial. Essas empresas operam com modelos complementares, como design de chips de alta performance, manufatura terceirizada de ponta e infraestrutura de armazenamento.

A Intel tem se descolado desse padrão de alta do setor devido a problemas estruturais próprios: perda market share para a AMD, atrasos crônicos na transição de suas litografias de fabricação e menor protagonismo inicial na infraestrutura de IA, o que reflete diretamente na estagnação ou queda de seus papéis em comparação aos concorrentes.

---

#Visão Geral do Código, Decisões e Dificuldades

##Decisões da Arquitetura
 - Pipeline de Dados Independente: O código baixa os dados via yfinance, trata valores ausentes e calcula os indicadores técnicos SMA, EMA e RSI, garantindo que o dataset esteja completo antes da modelagem.
 - Separação Temporal Estrita: A divisão de treino, validação e teste respeita a cronologia (train_end, val_end) para evitar vazamento de dados do futuro para o passado.
 - Arquitetura Híbrida (LSTM + GRU): O modelo PyTorch empilha camadas LSTM para capturar dependências de longo prazo e GRU para refinar a extração de características.
 - Delta Head: A camada final foi desenhada para calcular um "delta" (variação) que é somado ao último valor conhecido. Foi uma tentativa de forçar a rede a prever o movimento, embora a rede tenha otimizado esse delta para próximo de zero (causando o vício citado).

##Principais Dificuldades
A maior dificuldade no projeto é inerente à previsão de séries temporais financeiras. O ruído do mercado e a alta correlação entre o preço de t e t+1 fazem com que o gradiente descendente encontre na "cópia do dia anterior" o caminho de menor erro mais fácil, ignorando os indicadores técnicos gerados. O recalculo dinâmico dos indicadores na função predict_future também adicionou complexidade lógica.

---







[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tQHFU6lR)
