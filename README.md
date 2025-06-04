# Zero‑Shot Text‑Classification Cluster

This repository contains **scripts and templates** that let you spin‑up a multi‑GPU / multi‑host zero‑shot classification service.
All external traffic always hits **one public URL**; internally the requests are weighted across any number of LLM and **Geracl** model replicas.

---

## 1 · Configuration files

### `cluster.json`

Describe **which model runs on which GPU(s)** on each host.
Ranges are allowed (e.g. `cuda:1-3`).

For example:

```json
{
  "edge-01": {
    "cuda:0-1": "geracl",
    "cuda:2"  : "meta-llama/Llama-3.1-8B-Instruct"
  },
  "edge-02": {
    "cuda:0":   "qwen/Qwen1.5-4B"
  }
}
```

### `images.yaml`

Map **model names → Docker images** (can be private registries):

```yaml
# images.yaml
geracl: geracl-service:latest
Qwen/Qwen3-0.6B: llm-service:latest
```

### `weights.yaml` (optional)

Override backend **weights** in Nginx.
Defaults (if absent): geracl = 5, the rest = 1.

```yaml
meta-llama/Llama-3.1-8B-Instruct: 3
deepvk/GeRaCl-USER2-base: 2
```

---

## 2 · Generate deployment artifacts

```bash

python scripts/generate_compose.py   \
       --cluster cluster.json      \
       --images  images.yaml       \
       --base-port 8100            # gpu0→8100, gpu1→8101 …


python scripts/generate_nginx_conf.py \
       --cluster cluster.json        \
       --weights weights.yaml        \
       --mode per-container          \
   > nginx.conf
```

`generate_compose.py` drops files named `docker-compose.<host>.yml`.
`generate_nginx_conf.py` prints the full Nginx config to stdout.

---

## 3 · Deploy

### 3.1  Model hosts

```bash
# on every model host, e.g. edge‑01
scp docker-compose.edge-01.yml  edge-01:/opt/zero-shot/
ssh edge-01 "cd /opt/zero-shot && docker compose -f docker-compose.edge-01.yml up -d"
```

Repeat for every host defined in `cluster.json`.

### 3.2  Nginx front door

```bash
sudo cp nginx.conf /etc/nginx/nginx.conf
sudo nginx -t && sudo nginx -s reload
```

All public traffic now lands on **port 80** (or 443 if you terminate TLS).

---

## 4 · Calling the API (single endpoint)

Send every classification request to the **same URL** — Nginx will weight‑shift it to the fastest backend.

```bash
curl -X POST http://your-domain.com/classify \
     -H "Content-Type: application/json"      \
     -d '{
           "texts" : [
             "Утилизация катализаторов: как неплохо заработать",
             "Этот фильм мне очень понравился!"
           ],
           "labels": [
             ["экономика", "культура", "спорт"],
             ["позитивный", "нейтральный", "негативный"]
           ]
         }'
```

**Example response** (model‑agnostic):

```json
{
  "labels": ["экономика", "позитивный"],
}
```

The caller never needs to know which model or GPU produced each answer.
