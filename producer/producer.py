from kafka import KafkaProducer
import json
import time
import random

# Configuração do Kafka (ajuste o host se rodar em outro lugar depois)
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Simulador de dados baseado em medições de poços de petróleo
def generate_data():
    return {
        "well_id": f"WELL-{random.randint(1, 5):05}",
        "pressure": round(random.uniform(100, 300), 2),     # pressão (psi)
        "temperature": round(random.uniform(20, 90), 2),    # temperatura (°C)
        "flow_rate": round(random.uniform(1000, 5000), 2),  # vazão (bpd)
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

if __name__ == "__main__":
    print("Enviando dados simulados para o Kafka (Ctrl+C para parar)\n")
    try:
        while True:
            data = generate_data()
            producer.send("telemetry.raw", value=data)
            print(f"Enviado: {data}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        producer.close()
