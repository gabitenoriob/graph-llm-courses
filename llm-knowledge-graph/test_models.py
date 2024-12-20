from pathlib import Path

# Caminho padrão do cache
cache_path = Path.home() / ".cache" / "huggingface" / "transformers"
model_dirs = list(cache_path.glob("*"))

print("Modelos instalados localmente:")
for model_dir in model_dirs:
    print(model_dir.name)
