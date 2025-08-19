import re

def extrair(referencia):
  padrao = re.compile(r"(.+?)\s+\((\d{4})\)\.\s+(.+?)\.\s+(.+?)\.")
  match = padrao.match(referencia)

  if match:
    autores = [autor.strip() for autor in re.split(r",\s*| & ", match.group(1))]
    ano = match.group(2)
    titulo = match.group(3)
    editora = match.group(4)

    return {
        "autores": autores,
        "ano": ano,
        "titulo": titulo,
        "editora": editora
    }
  else:
    return None

referencia_usuario = input("digite a ref: ")
""" Exemplos
# Manning, C. D., Manning, C. D., & Schutze, H. (1999). Foundations of statistical natural language processing. MIT press.
# Russell, S. J., & Norvig, P. (2010). Artificial intelligence: A modern approach. Prentice Hall.
#Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
# Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: Data mining, inference, and prediction. Springer.
# Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
"""
info = extrair(referencia_usuario)

if info:
  print("Autores:", ", ".join(info["autores"]))
  print("Ano de publicação:", info["ano"])
  print("Título:", info["titulo"])
  print("Editora:", info["editora"])
else:
  print("https://shre.ink/tzvU")