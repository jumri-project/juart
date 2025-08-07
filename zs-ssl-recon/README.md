# ZS-SSL: Zero-Shot Self-Supervised Learning

## Autoformatting

```
isort zs_ssl_recon --profile black
black zs_ssl_recon
```

## Sync to Voila

```
sudo rsync -av --delete /home/jovyan/zs-ssl-recon/zs_ssl_recon /home/software/modules/zs-ssl-recon
sudo rsync -av --delete /home/jovyan/zs-ssl-recon/schemes /home/software/modules/zs-ssl-recon
sudo rsync -av --delete /home/jovyan/zs-ssl-recon/dashboards/ /home/software/dashboards/zs-ssl-recon
```

## Sync to JUST

```
sudo rsync -i ~/.ssh/id_ed25519 -avz /home/jovyan/zs-ssl-recon zimmermann9@judac.fz-juelich.de:/p/project1/drecon/qrage-dl/
```
