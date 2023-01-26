import sqlite3

def connectDB(dbname):
    # dbname : string. The name of database file (.db) which you want to connect.
    #          This function will create the database file if it does not exist. 
    conn = sqlite3.connect(dbname)
    return conn



