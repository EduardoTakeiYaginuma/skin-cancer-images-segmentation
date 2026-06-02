# Handoff — Projeto MLOps Skin Cancer (para a IA do par do Gabriel)

> Documento para passar contexto pra outra IA continuar de onde paramos. Gabriel é aluno de MLOps no Insper, projeto final em par. Entrega final: **5 de junho de 2026**. Hoje é **2 de junho de 2026**.

---

## Objetivo

Validar que o projeto **skin-cancer-images-segmentation** (classificação binária de melanoma com ResNet50 + U-Net de segmentação) está alinhado com a [rúbrica MLOps do Insper](https://insper.github.io/mlops/project/project/) para garantir **conceito A**, e depois gravar um vídeo explicativo de 3-5 minutos.

### Rúbrica resumida
- **C** = todos os 7 itens base (inclui **vídeo de 3-5 min** explicativo)
- **B** = C + 2 itens da lista B
- **A** = C + 4 ou mais itens da lista B

### Status do projeto contra a rúbrica
Auditoria completa do código mostra **5/5 itens da lista B implementados** (MLflow, deploy automatizado, IaC com CloudFormation, monitoramento de drift, retrain trigger). Falta só gravar o vídeo (requisito de C) e idealmente confirmar deploy ativo na AWS.

---

## O que JÁ foi testado e está ✅ funcionando

### Nível 1 — Estrutural
| Validação | Resultado |
|---|---|
| `ruff check .` | All checks passed |
| `python3 -m compileall` em todos os módulos | OK |
| `bash -n scripts/*.sh` | OK (deploy_lambda.sh, dvc_setup_s3.sh) |
| YAML lint (dvc.yaml, params.yaml, docker-compose.yml, .github/workflows/ci.yml) | Todos válidos |
| `cfn-lint infra/cloudformation.yaml` | Sem erros bloqueantes (apenas avisos sobre regiões AWS novas e 1 info opcional sobre `${AWS::Partition}`) |

### Nível 2 — Testes unitários
```
.venv/bin/pytest tests/ -v --tb=short
→ 15 passed, 2 skipped em 27s
```
- `test_api.py`: 3/3 passed (health + predict + content-type)
- `test_preprocessing.py`: 12/12 passed (bbox, normalize, pad_to_square, conversões PIL)
- `test_feature_store.py`: 1/2 passed (o que pula precisa de `data/processed/treated_manifest.csv` gerado pelo notebook 03)

### Nível 4 — Feature Store
- `python feature_store/scripts/prepare_sources.py` rodou e gerou `feature_store/data/sources/lesion_classification.parquet` (10.015 registros, schema OK).
- O segundo parquet (`preprocessing_stats`) precisa do notebook 03_preprocessing rodando com imagens reais — pulado.
- `feast apply` não rodou por conflito de versões (feast 0.42.0 exige `protobuf<5` e `pyarrow<18.1`, incompatível com o resto do venv). **No CI funciona** porque usa `pip install feast==0.42.0 --no-deps`. Para reproduzir localmente, recomendado rodar dentro do container.

### Nível 5 — Monitoramento (drift + retrain)
```
.venv/bin/python -m monitoring.run_monitoring --shift 0.25
→ KS test melanoma_prob: stat=0.7490, p=0.0000, drift=True
→ Chi-Square triage_zone: stat=205.13, p=0.0000, drift=True
→ ⚠ DRIFT DETECTADO em: melanoma_prob_ks, triage_zone_chi2
→ [DRY RUN] Re-treino não executado.
```
Artefatos gerados em `monitoring/reports/20260602_103243/`:
- `drift_summary.json` (estatísticas + p-values + contagens por categoria)
- `melanoma_prob_dist.png` e `triage_zone_dist.png` (visualizações comparativas)

O retrain trigger logou corretamente em dry-run.

### Nível 6 — Docker e API local
- `docker build -f Dockerfile -t melanoma-api:dev .` → sucesso (~5 min)
- `docker compose up -d api` → container subiu, porta 8000 mapeada
- `GET /docs` → 200 OK
- `GET /openapi.json` → 200 OK
- `GET /health` e `POST /predict` → **500 esperado**: `FileNotFoundError: /app/outputs/models/model_comparison/resnet50_aug_224x224.pt`. Os modelos estão no DVC remote (S3), não no filesystem local. Os testes unitários já cobrem `/predict` com predictor mockado, então o código está validado.
- Container foi derrubado com `docker compose down`.

---

## O que está ⚠️ pendente / não testado

### Nível 3 — DVC pipeline end-to-end
**Bloqueio:** dataset ISIC 2020 (10k imagens + máscaras) e checkpoints estão no DVC remote (S3) — `dvc pull` precisa de credenciais AWS.

Para destravar:
```bash
.venv/bin/pip install "dvc[s3]"
dvc pull           # baixa images/, masks/, metadata.csv, outputs/models/
dvc repro evaluate # só avaliação, não re-treina
```

### Deploy AWS ao vivo — **SITUAÇÃO COMPLICADA**

Encontramos na AWS (Account `177660460967`, profile `mlops`, SSO Insper):
- Em `us-east-2`: Lambdas `aps03tf_predict_gabrielfmm` e `aps03_predict_gabrielfmm` + API Gateway `https://1shdwk4os2.execute-api.us-east-2.amazonaws.com`
- **MAS ao testar com payload, a resposta foi:**
  ```
  HTTP 400
  {"error": "Missing required fields: age, job, marital, education, balance, housing, duration, campaign"}
  ```
- Esses campos são do **dataset UCI Bank Marketing** — não é o projeto de melanoma! Essa Lambda é leftover de **outra APS** do Gabriel.

**A Lambda do projeto de melanoma é (provavelmente) do colega de par, não do Gabriel.** Buscamos em 6 regiões filtrando por "melanoma|skin|cancer|mlops" e não achamos nada. Sem o nome correto ou o username do par, não dá pra localizar.

### O que precisa ser feito para confirmar o deploy
**Você (IA do par)** está na máquina/sessão de quem **fez o deploy**, então provavelmente é mais fácil:
1. Listar as Lambdas que o par criou:
   ```bash
   for R in us-east-1 us-east-2 us-west-2 sa-east-1; do
     echo "=== $R ==="
     aws lambda list-functions --profile mlops --region $R \
       --query 'Functions[].[FunctionName,LastModified]' --output table
   done
   ```
2. Identificar a Lambda do melanoma (padrão de nome no Insper: `aps<num>_predict_<usernameparcial>` ou `tf_predict_...`).
3. Pegar a URL do API Gateway:
   ```bash
   aws apigatewayv2 get-apis --profile mlops --region <REGION> \
     --query 'Items[].[Name,ApiEndpoint,ApiId]' --output table
   ```
4. Testar com a imagem de exemplo `IMG_5115.jpg` no raiz do repo:
   ```bash
   # O lambda_handler.py atual espera raw bytes da imagem:
   curl -X POST "<API_URL>/predict?filename=test.jpg" \
     -H "Content-Type: image/jpeg" \
     --data-binary @IMG_5115.jpg
   ```
5. Se a versão deployada for antiga (commit anterior a 2c5b300 de 12-mai), pode estar esperando outro formato — checar os logs:
   ```bash
   aws logs describe-log-streams --profile mlops --region <REGION> \
     --log-group-name /aws/lambda/<FUNC_NAME> \
     --order-by LastEventTime --descending --max-items 1
   ```

### Nível 8 — CI no GitHub
Não rodamos `git push` pra não mexer no shared state. **Recomendado antes de gravar:**
- Push pra `mlOps_project` e conferir no GitHub Actions que lint + test + docker-build ficam verdes (visual pro vídeo)

---

## Estado atual do repositório (commit local)

- Branch: `mlOps_project`
- Último commit: `88084ad Add initial retrain log entries for drifted features`
- Working tree: clean
- `.venv` local com Python 3.12, PyTorch 2.5.1 CPU, deps de teste instaladas

### Estrutura
```
api/                  FastAPI (main, schemas, dependencies)
skin_app/             core (inference, logging_config)
monitoring/           drift_detector, retrain_trigger, run_monitoring, reports/
feature_store/        Feast (entities, views, services + scripts)
infra/                cloudformation.yaml
.github/workflows/    ci.yml (lint, test, docker-build, cron monitoring semanal)
scripts/              deploy_lambda.sh, dvc_setup_s3.sh
tests/                test_api.py, test_preprocessing.py, test_feature_store.py
docs/                 project_report.md, modeling_2_training_journal.md
notebooks/            EDA, preprocessing, augmentation
data/                 .dvc pointers (real data no S3)
outputs/models/       .dvc pointers (modelos no S3)
```

### Stack MLOps
- **Treino**: PyTorch ResNet50 (classificação binária melanoma vs não-melanoma) + U-Net (segmentação)
- **Tracking**: MLflow com Model Registry e alias `production` (promoção condicional se AUC ≥ 0.85)
- **Versionamento**: DVC (pipeline + remote S3) + Feast (feature store)
- **Serving**: FastAPI local + AWS Lambda (image-based via ECR) + API Gateway HTTP v2
- **IaC**: CloudFormation parametrizado
- **Monitoramento**: scipy (KS + Chi²) substituiu Evidently (commit 5d2f3ea), com retrain via `dvc repro --force`
- **CI**: GitHub Actions (lint, test, docker-build, cron semanal de monitoring)

---

## Script do vídeo (3-5 min)

Já gerado, segue o resumo dos blocos:

1. **[0:00–0:25]** Abertura: problema (triagem de melanoma), dataset ISIC 2020, foco em operacionalização
2. **[0:25–0:55]** Pipeline de dados: DVC com 3 stages (preprocess/train/evaluate), versionamento via S3
3. **[0:55–1:25]** Feature Store com Feast: 2 views (classificação + stats de preprocessing), online/offline
4. **[1:25–2:00]** Treino + MLflow: tracking completo, Model Registry, promoção automática
5. **[2:00–2:40]** Deployment: Lambda + ECR + CloudFormation + FastAPI + Streamlit
6. **[2:40–3:20]** Monitoramento: drift KS/Chi², retrain automático via DVC
7. **[3:20–3:50]** CI/CD: GitHub Actions com cron semanal
8. **[3:50–4:15]** Fechamento e resumo dos componentes

Versão completa do script está no chat anterior (gerar de novo se necessário).

---

## Pendências para fechar antes de gravar

| # | Item | Prioridade | Comentário |
|---|---|---|---|
| 1 | Achar a Lambda real do projeto (deploy do par) | **Alta** | Sem ela, o vídeo não consegue mostrar inferência ao vivo na AWS |
| 2 | Testar `POST /predict` com `IMG_5115.jpg` no endpoint correto | Alta | Smoke test final |
| 3 | `git push` pra ver CI verde no GitHub | Média | Bom pra mostrar no vídeo |
| 4 | (Opcional) `dvc pull` + `dvc repro evaluate` localmente | Baixa | Só se quiser mostrar pipeline rodando ao vivo |
| 5 | Gravar o vídeo | **Crítico** | É requisito de C — sem ele cai pra D |

---

## Resumo executivo para retomar

**Onde paramos:** Já validamos tudo que é validável local (testes, lint, drift, docker, API). O único item bloqueante é encontrar e testar o **deploy ativo da Lambda na AWS** que foi feito pelo **colega de par do Gabriel** — não pelo Gabriel. As Lambdas com nome do Gabriel (`aps03tf_predict_gabrielfmm`) são de outra APS (bank marketing), não desta.

**Próxima ação imediata para a IA do par:** rodar `aws lambda list-functions --profile mlops --region us-east-2 --output table` (e outras regiões), identificar a Lambda do melanoma, testar `POST /predict` com a imagem `IMG_5115.jpg`, e confirmar que retorna JSON com `melanoma_prob`, `triage_zone`, etc.

Depois disso, gravar o vídeo seguindo o script e entregar antes de 5 de junho.
