import yfinance as yf
import psycopg2,sys
import pandas as pd

# Ticker symbol for Alphabet Inc. (Google) is 'GOOGL'
def db_connect():
        # Database connection details
    db_host = "localhost"
    db_port = "5432"
    db_name = ""
    db_user = "postgres"
    db_password = ""

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )
    return conn

def load_ticker(ticker='GOOGL',drop_and_recreate=0):

    # Create a cursor object
    conn = db_connect()
    cur=conn.cursor()
    if drop_and_recreate:
        cur.execute("drop table stock;")      
        create_table_query = """
            CREATE TABLE stock (
                date DATE PRIMARY KEY,
                ticker varchar(50),
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC,
                volume BIGINT,
                dividends NUMERIC,
                stock_splits NUMERIC
            );
            """

            # Execute the query to create the table
        cur.execute(create_table_query)  
        

    cur.execute("SELECT COUNT(*) FROM stock s where s.ticker='%s';"%ticker)
    # Fetch and print the count
    record_count = cur.fetchone()[0]
    print('No. of record in table',record_count)
    if int(record_count) > 0:
        print( "stock price available")
        return
        
    # Get stock data
    google_stock = yf.Ticker(ticker)

    # Fetch historical market data for the past 1 year
    hist = google_stock.history(period="1y")

    # Insert data from the 'hist' DataFrame into the 'stock' table
    insert_query = """
    INSERT INTO stock (date, ticker,open, high, low, close, volume, dividends, stock_splits)
    VALUES (%s, %s,%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (date) DO NOTHING;
    """

    # Prepare the data for insertion
    for index, row in hist.iterrows():
        cur.execute(insert_query, (
            index,  # date is the index
            ticker,
            row['Open'],
            row['High'],
            row['Low'],
            row['Close'],
            row['Volume'],
            row['Dividends'],
            row['Stock Splits']
        ))

    # Commit the changes and close the connection
    conn.commit()
    count_query = "SELECT COUNT(*) FROM stock where ticker='%s';"%(ticker)
    cur.execute(count_query)
    # Fetch and print the count
    record_count = cur.fetchone()[0]
    print(f"Number of records in the 'stock' table: {record_count}")

    #get latest stock price
    
    cur.close()
    conn.close()

    print("Data loaded successfully into the 'stock' table.")

# The standard boilerplate to call the main() function

def main():
    try:
        for ticker in ['NVDA','GOOG']:    
            load_ticker(ticker=ticker)
    except Exception as ex:
        print(str(ex))

if __name__ == "__main__":
    main()

