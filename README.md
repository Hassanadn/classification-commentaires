# Classification de Commentaires avec BERT & FastAPI

Ce projet vise à développer une application de classification de commentaires (analyse de sentiments ou modération automatique) en combinant des techniques de NLP avec un modèle BERT, le tout intégré dans une API web via FastAPI. Il intègre également du monitoring via Prometheus, Grafana et Weights & Biases (wandb).

---

## 📌 Objectifs

- Nettoyer et traiter des données de commentaires.
- Entraîner un modèle de classification basé sur BERT.
- Déployer une API RESTful pour interagir avec le modèle.
- Visualiser les résultats et surveiller les performances en temps réel.
- Assurer la reproductibilité avec DVC, Docker, et wandb.

---

## Architecture du projet

### Schéma visuel

![Architecture du projet](./docs/"Project Architecture".png)


Application web permettant de classifier automatiquement les commentaires des utilisateurs comme positifs ou négatifs.

## grafana +prometheus
commande: docker-compose up -d
grafana: http://localhost:3000
prometheus: http://localhost:9090
compte_grafana:   nom:admin  code:admin

## Installation

1. Cloner le dépôt
```bash
git clone https://github.com/Hassanadn/classification-commentaires.git
cd classification-commentaires
