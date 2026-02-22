# [M2] Laboratorio - Construcción Directa de AFD y ecosistema de reconocimiento de expresiones regulares

## Cómo ejecutar

```bash
python lab2.py
```

Al iniciar, ingrese una expresion regular `r`. Luego puede ingresar cadenas `w`.

- `exit` termina el programa.
- `new` permite ingresar otra expresion regular.

El programa imprime el AFD por construccion directa y el AFD minimizado.

- `afd_directa.dot`
- `afd_min.dot`


## Ejemplos
```text
r = (a|b)*abb
    w = ACEPTADAS
    abb
    aabb
    babb
    abababb
    bbbbbbbbbbbbabb
    aaaaaaaaaaaaabb
    abababababababb

    w = RECHAZADAS
    ab
    aba
    abba
    ababababa
    bbbbbbbb
    aaaaaaaa
    abbbbbbbb

r = (a|b)*(aa|bb)(a|b)*
    w = ACEPTADAS
    aa
    bb
    aab
    baa
    abba
    bbaaab
    abababbaabab
    bbbbbbbbbbbb
    aaaaaaaaaaaa

    w = RECHAZADAS
    a
    b
    ab
    ba
    abababab
    bababa

r = (a|b)*a(a|b)(a|b)(a|b)
    w = ACEPTADAS
    aaaa
    baaaa
    ababaaaa
    bbbbbaaaa
    ababababaaaa

    w = RECHAZADAS
    bbb
    aa
    bbbb
    bbaa
    babababababa

r = (a|b|c)*abc(a|b|c)*
    w = ACEPTADAS
    abc
    aabc
    abcc
    abcabc
    bbbbabcbbbb
    ccccabcaaaa
    ababababcababab

    w = RECHAZADAS
    ab
    acb
    bac
    aaaaaaa
    bbbbbbb
    ccccccc

```

## Referencias

- Hopcroft, Motwani, Ullman. *Introduction to Automata Theory, Languages, and Computation* (3rd ed.). Minimization de AFD.
- Poe, V. A. (n.d.). Autómata Finito Determinista en python. Stack Overflow En Español. https://es.stackoverflow.com/questions/76447/aut%C3%B3mata-finito-determinista-en-python
- Pagina nueva 1. (n.d.). https://www.profesores.frc.utn.edu.ar/sistemas/ssl/marciszack/ghd/T-M-AFD.htm.
- OpenAI. (2024). ChatGPT (GPT-4/GPT-5 family) [Large language model].
https://chat.openai.com/