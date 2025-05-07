# from influxdb import InfluxDBClient

# def log_metric_to_influx(measurement, fields, tags=None):
#     client = InfluxDBClient(host='localhost', port=8086, database='metrics')
#     json_body = [{
#         "measurement": measurement,
#         "tags": tags or {},
#         "fields": fields
#     }]
#     client.write_points(json_body)
#     client.close()

# # Exemple d'utilisation
# if _name_ == "_main_":
#     log_metric_to_influx(
#         measurement="model_performance",
#         tags={"model": "v1"},
#         fields={"accuracy": 0.92, "f1_score": 0.88}
#     )