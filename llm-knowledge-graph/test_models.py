from pathlib import Path

# Caminho padrão do cache
cache_path = Path.home() / ".cache" / "huggingface" / "hub"
model_dirs = list(cache_path.glob("*"))

if model_dirs:
    print("Modelos instalados localmente:")
for model_dir in model_dirs:
    print(model_dir.name)

if not model_dirs:
    print("Não há modelos instalados")
