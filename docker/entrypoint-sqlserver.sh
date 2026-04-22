#!/bin/bash
# Start SQL Server, wait for it, run init.sql

/opt/mssql/bin/sqlservr &
SQL_PID=$!

echo "Waiting for SQL Server to start..."
for i in {1..30}; do
    /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" \
        -C -Q "SELECT 1" > /dev/null 2>&1 && break
    sleep 2
done

echo "Running init.sql..."
/opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" \
    -C -i /docker-entrypoint-initdb.d/init.sql

wait $SQL_PID
