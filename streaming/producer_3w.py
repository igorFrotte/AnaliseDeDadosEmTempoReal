from kafka import KafkaProducer
import json
import time
import pandas as pd
import random

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "telemetry.raw"

# Caminho relativo a partir da pasta onde o script estiver
CSV_PATH = "../data/petrobras_3w_dados.csv"

def carregar_dados():
    # Carrega apenas as colunas relevantes + classe (opcional) + timestamp se existir
    df = pd.read_csv(CSV_PATH)

    # Garante que as colunas existem
    colunas_necessarias = [
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "classe",
    ]

    for c in colunas_necessarias:
        if c not in df.columns:
            raise ValueError(f"Coluna {c} não encontrada no CSV!")

    # Se houver coluna de timestamp real no CSV e quiser usar, adapte aqui.
    # Por simplicidade, vamos usar apenas as colunas acima e gerar o timestamp na hora do envio.
    return df[colunas_necessarias]

if __name__ == "__main__":
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    df = carregar_dados()
    print(f"{len(df)} linhas carregadas do CSV 3W.")

    try:
        print("Enviando linhas reais do 3W para o Kafka (Ctrl+C para parar)...")
        while True:
            # Escolhe uma linha aleatória
            row = df.sample(1).iloc[0]

            msg = {
                # features do modelo
                "P-TPT": float(row["P-TPT"]),
                "T-TPT": float(row["T-TPT"]),
                "P-MON-CKP": float(row["P-MON-CKP"]),
                "T-JUS-CKP": float(row["T-JUS-CKP"]),
                "P-JUS-CKGL": float(row["P-JUS-CKGL"]),

                # rótulo verdadeiro (opcional, só para debug no consumer)
                "classe_real": int(row["classe"]),

                # timestamp de envio
                "timestamp_envio": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            producer.send(TOPIC, value=msg)
            print(f"Enviado: {msg}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        producer.flush()
        producer.close()
