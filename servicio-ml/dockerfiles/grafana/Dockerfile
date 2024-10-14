# Usa la imagen base oficial de Grafana
FROM grafana/grafana:latest

# Copia las configuraciones de provisioning
COPY ./provisioning /etc/grafana/provisioning

# Copia los dashboards
# base en dashboards oficiales de minio
COPY ./dashboards/minio /var/lib/grafana/dashboards/minio
# https://github.com/data-burst/airflow-monitoring-and-alerting/tree/main/config_files/grafana/var/lib/grafana/dashboards
COPY ./dashboards/airflow /var/lib/grafana/dashboards/airflow
# https://grafana.com/grafana/dashboards/14114-postgres-overview/
COPY ./dashboards/postgres /var/lib/grafana/dashboards/postgres