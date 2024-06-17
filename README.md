# Disciplina de Green AI, Atividade 2: WiSARD

Este repositório é uma atividade prática de implementação de modelos de WiSARD para classificação. 

Os conjuntos de dados utilizados foram fornecidos pelo repositório da UCI Machine Learning Repository.

Dentre os já utilizados da última atividade foram adicionados os seguintes conjuntos de dados:

- glass: https://archive.ics.uci.edu/dataset/42/glass+identification
- hepatitis: https://archive.ics.uci.edu/dataset/571/hcv+data

## Rodando e averiguando os resultados:

Para rodar os exemplos deste código, certifique-se que o diretório atual está na pasta principal do projeto (wisard_grai, se o diretório não foi renomeado) e use o comando:

`python src/main.py --dataset <nome_do_dataset>`

A opção de dataset é obrigatória, use o nome de um dos datasets suportados pelo código: abalone, internet, soybean, glass e hepatitis.

Esta implementação salva os resultados da classificação automaticamente na pasta results, com um arquivo csv diferente para cada dataset. Os resultados de uma primeira rodada de testes com cada dataset já está disponível.