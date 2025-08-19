def verificar_senha_forte(senha):
    if len(senha) < 8:
        return False

    maiscula = False
    for letra in senha:
        if letra >= 'A' and letra <= 'Z':
            maiscula = True
            break

    if not maiscula:
        return False

    minuscula = False
    for letra in senha:
        if letra >= 'a' and letra <= 'z':
            minuscula = True
            break

    if not minuscula:
        return False

    digito = False
    for caractere in senha:
        if caractere >= '0' and caractere <= '9':
            digito = True
            break

    if not digito:
        return False

    caracteresEspeciais = "!@#$%&*"
    temEspecial = False
    for caractere in senha:
        if caractere in caracteresEspeciais:
            temEspecial = True
            break

    if not temEspecial:
        return False

    return True

senhas_fracas = [
    "123",
    "password",
    "Password",
    "Password1",
    "PASSWORD1!",
    "password1!"
]

print("SENHAS FRACAS:")
for senha in senhas_fracas:
    resultado = verificar_senha_forte(senha)
    print(f"'{senha}' -> {resultado}")
print()

senhas_fortes = [
    "Password1!",
    "MinhaSenh@123",
    "AbC123#def",
    "Teste2024!",
    "Segura&456"
]

print("SENHAS FORTES:")
for senha in senhas_fortes:
    resultado = verificar_senha_forte(senha)
    print(f"'{senha}' -> {resultado}")
print()

# o teste
senha_usuario = input("digite sua senha: ")
if verificar_senha_forte(senha_usuario):
    print("True")
else:
    print("False")
