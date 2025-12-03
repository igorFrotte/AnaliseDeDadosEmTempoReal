AnaliseDeDadosEmTempoReal

O que é este projeto

“AnaliseDeDadosEmTempoReal” é uma aplicação em Python voltada para ingestão, processamento e visualização de dados em tempo real utilizando SparkML.

Objetivo

-   Ler dados continuamente de fontes de streaming.
-   Processar e agregar dados.
-   Exibir visualizações dinâmicas via dashboard.

Tecnologias

-   Python 3.x
-   Bibliotecas em requirements.txt
-   Docker 

Estrutura

/streaming — ingestão de dados
/train — ptreinamento e modelo treinado
dashboard_3w.py — dashboard
docker-compose.yml — containerização
requirements.txt — dependências

Como executar

1.  Clone: git clone
    https://github.com/igorFrotte/AnaliseDeDadosEmTempoReal
2.  Instale dependências: pip install -r requirements.txt
3.  Suba o container
4.  Execute o producer e o consumer
5.  Execute: python3 dashboard_3w.py

Funcionalidades

-   Streaming de dados
-   Processamento em tempo real
-   Dashboard dinâmico
