"""
Registra as definições de features (entities, views, services) no registry local do Feast.

Execute após prepare_sources.py e a cada mudança em feature_repo/:
    python feature_store/scripts/apply_registry.py
"""

import subprocess
import sys
from pathlib import Path
import shutil

REPO_PATH = Path(__file__).resolve().parent.parent / "feature_repo"


def main() -> None:
    if not REPO_PATH.exists():
        print(f"Erro: feature_repo não encontrado em {REPO_PATH}")
        sys.exit(1)

    print(f"Aplicando registry em: {REPO_PATH}")
    feast_executable = Path(sys.executable).with_name("feast")
    if feast_executable.exists():
        feast_cmd = [str(feast_executable), "apply"]
    else:
        feast_on_path = shutil.which("feast")
        if feast_on_path is None:
            print(
                "Erro: CLI do Feast não encontrada.\n"
                "Ative o ambiente virtual do projeto ou instale o pacote nele."
            )
            sys.exit(1)
        feast_cmd = [feast_on_path, "apply"]

    result = subprocess.run(
        feast_cmd,
        cwd=str(REPO_PATH),
        capture_output=False,
    )

    if result.returncode != 0:
        print("\nErro ao aplicar o registry. Verifique se:")
        print(f"  1. Feast está instalado no ambiente de {sys.executable}")
        print("  2. Os arquivos Parquet foram gerados: python3 feature_store/scripts/prepare_sources.py")
        sys.exit(result.returncode)

    print("\nRegistry aplicado com sucesso.")
    print("Próximo passo: python3 feature_store/scripts/get_historical_features.py")


if __name__ == "__main__":
    main()
