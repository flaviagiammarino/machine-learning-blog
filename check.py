
import pandas as pd
import clickhouse_connect

clickhouse_client = clickhouse_connect.get_client(
    host="<clickhouse-host>",
    user="<clickhouse-user>",
    password="<clickhouse-password>",
    secure=True
)

df = clickhouse_client.query_df(
    """
    select
        timestamp,
        total_load
    from
        total_load_data
    where
        timestamp >= toDateTime('2025-08-18 23:45:00') - INTERVAL 14 DAYS
    order by
        timestamp asc
    """
)

output = pd.merge(
    left=df,
    right=pd.concat([predictions, forecasts], axis=0),
    on="timestamp",
    how="outer"
)