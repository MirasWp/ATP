import mysql.connector
from mysql.connector import Error

def conectar():
    try:
        conexao2 = mysql.connector .connect(
            host = "localhost",
            user = "root",
            password = 'password',
            database ="ATP3"
        )
        if conexao2.is_connected():
            print ("deu bom")
            return conexao2

    except Error as e:
        print("erro ao conectar ao banco de dados", e)
        return None
conexao = conectar()