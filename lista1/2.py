import re

def valida(email):
    padrao = r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(padrao, email) is not None

# exemplos
testes = [
    "usuario@exemplo.com",
    "test.email@gmail.com",
    "email_sem_arroba.com",
    "usuario@dominio.c"
]

for email in testes:
    print(f"{email} -> {valida(email)}")

while True:
    usermail = input("digite seu email: ")

    if usermail == "0": # para encerrar o programa
      print("programa encerrado")
    break

    if valida(usermail):
        print("True")
    else:
        print("False")