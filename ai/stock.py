import yfinance as yf
import psycopg2,sys
import pandas as pd

# Ticker symbol for Alphabet Inc. (Google) is 'GOOGL'

def db_sqlalchemy():
    db_host = "localhost"
    db_port = "5432"
    db_name = ""
    db_user = "postgres"
    db_password = ""
    
    from sqlalchemy import create_engine
    #conn = create_engine('postgresql+psycopg2://postgres:@localhost:5432/')
    connection_string = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    conn = create_engine(connection_string)
    return conn

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

def predict_linear_reg(ticker=None):
    import pandas as pd
    import yfinance as yf
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    # Load historical stock data for Google (Alphabet Inc.)
    ticker = ticker
    #data = yf.download(ticker, start='2010-01-01', end='2024-10-20')

    query = "SELECT close as \"Close\" FROM stock where ticker='%s' order by date ;"%ticker
    print(query)
    # Step 3: Read the SQL query results into a Pandas DataFrame
    conn=db_sqlalchemy()
    data = pd.read_sql_query(query, conn)

    data = data[['Close']]
    data.reset_index(inplace=True)

    # Create features and labels
    data['Prev Close'] = data['Close'].shift(1)
    data.dropna(inplace=True)
    X = data[['Prev Close']].values  # Features
    y = data['Close'].values          # Labels

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))  # 1 input feature
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, verbose=0)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Model Loss: {loss}')

    # Make predictions
    predictions = model.predict(X_test)
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
    print(results.head())

def predict_stock(ticker=None):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from matplotlib import pyplot as plt

    # Step 1: Load the dataset
    # Step 2: Write the SQL query
    query = "SELECT close FROM stock where ticker='%s' order by date ;"%ticker
    print(query)
    # Step 3: Read the SQL query results into a Pandas DataFrame
    conn=db_sqlalchemy()
    data = pd.read_sql_query(query, conn)
    print(data.shape)
    print(len(data))
    #data = pd.read_csv('GOOGL_stock_data.csv')  # Replace with the correct path to your dataset

    # Step 2: Preprocess the data (Normalize the data)
    # We will use only the 'Close' price for simplicity
    data = data['close'].values
    data = data.reshape(-1, 1)

    # Normalize the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Step 3: Create training and test sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Create sequences (look-back) to train the LSTM
    def create_sequences(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60  # Look back for 60 days to predict the next price
    X_train, y_train = create_sequences(train_data, time_step)
    X_test, y_test = create_sequences(test_data, time_step)

    # Reshape data to be [samples, time steps, features] which is required for LSTM
    #X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Step 4: Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 5: Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=10)

    # Step 6: Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    return
    # Invert the scaling to get actual prices
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_actual = scaler.inverse_transform([y_train])
    y_test_actual = scaler.inverse_transform([y_test])

    # Step 7: Plot the results
    # Shift train predictions for plotting
    train_predict_plot = np.empty_like(scaled_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

    # Shift test predictions for plotting
    test_predict_plot = np.empty_like(scaled_data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

    # Plot baseline and predictions
    plt.figure(figsize=(10, 6))
    plt.plot(scaler.inverse_transform(scaled_data), label='Actual Stock Price')
    plt.plot(train_predict_plot, label='Train Predict')
    plt.plot(test_predict_plot, label='Test Predict')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def main():
    try:
        for ticker in ['GOOG']:  

            #load_ticker(ticker=ticker)
            #predict_stock(ticker=ticker)
            predict_linear_reg(ticker=ticker)

        #delta_change_ticker(t)
    except Exception as ex:
        import traceback
        print(traceback.print_exc())
        print(str(ex))

if __name__ == "__main__":
    main()

