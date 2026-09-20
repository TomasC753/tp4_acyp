= Objetivos
El objetivo principal de esta actividad consiste en aplicar los conocimientos adquiridos en el módulo cuatro de la asignatura. Específicamente se abordará el concepto de paralelismo mediante el uso de cómputo paralelo en la nube. La actividad debe ser llevada a cabo de manera individual.

En concreto los objetivos de esta práctica son:
- Uso de la plataforma GitHub Codespaces para efectuar cómputo paralelo.
- Comprender conceptos de paralelismo.
- Aprender algunos algoritmos de machine learning que se pueden paralelizar.

= Situación problemática
Para adquirir experiencia en computo paralelo es clave comprender como se implementan los algoritmos en paralelo. Ademas es importante conocer algunos ámbitos de aplicación de los algoritmos paralelos, como por ejemplo el calculo matemático y el machine learning.

= Consignas

1. Analizar e interpretar las salidas por consola.
2. Responder las siguientes preguntas en forma breve: dos renglones como máximo.

== Producto de matrices
+ ¿Qué función cumple MPI_Scatter en el código?
+ ¿Por qué se usa MPI_Bcast para la matriz B?
+ ¿Cuántas filas calcula cada proceso cuando se usan 4 procesos para una matriz 4x4?
+ ¿Qué ocurre si se ejecuta con más procesos que filas de la matriz?
+ ¿Por qué solo el proceso 0 imprime el resultado final?

== Preguntas: Montecarlo π
+ ¿Por qué cada proceso necesita una semilla aleatoria diferente?
+ ¿Qué función MPI permite sumar los aciertos de todos los procesos?
+ Si ejecutas con 1 y con 4 procesos, ¿el valor de π es exactamente igual? ¿Por qué?
+ ¿Cómo afecta el número de procesos al tiempo de ejecución?
+ ¿Podrías modificar el programa para que cada proceso lance distinta cantidad de dardos?

== Preguntas: K-means
+ ¿Qué diferencia hay entre MPI_Reduce y MPI_Allreduce? ¿Por qué aquí usamos MPI_Allreduce?
+ Si el número de procesos no divide exactamente a N_PUNTOS, ¿cómo podrías distribuir los datos?
+ ¿Por qué los centros iniciales se eligen aleatoriamente? ¿Qué riesgo existe?
+ ¿Qué papel cumple MPI_Bcast dentro del bucle principal?
+ Al aumentar MAX_ITER o la cantidad de puntos, ¿cómo se comporta la relación cómputo/comunicación?

= Desarrollo

== Preparación del entorno de trabajo

== Analisis de salidas

=== Analisis de la salida del programa de multiplicación de matrices
En la terminal, se posicionó sobre el directorio `./ejemplos_base` y se ejecutó
```bash
mpic++ matmul.cpp -o matmul_cpp && mpirun -np 4 ./matmul_cpp
```
Para compilar y ejecutar el programa de multiplicación de matrices en c++. La salida por consola fue la siguiente: 

```txt
Matriz A generada por proceso 0:
5 6 5 8 
1 5 4 2 
9 0 9 4 
0 1 0 8 

Matriz B generada por proceso 0:
7 2 0 7 
3 0 5 8 
3 4 7 8 
8 5 8 5 

Proceso 0 - Fila recibida de A: 5 6 5 8

Proceso 1 - Fila recibida de A: 1 5 4 2

Proceso 2 - Fila recibida de A: 9 0 9 4

Proceso 3 - Fila recibida de A: 0 1 0 8

Proceso 0 - Resultado parcial (fila de C): 132 70 129 163

Proceso 1 - Resultado parcial (fila de C): 50 28 69 89

Proceso 2 - Resultado parcial (fila de C): 122 74 95 155

Proceso 3 - Resultado parcial (fila de C): 67 40 69 48


Matriz resultado C:
132 70 129 163 
50 28 69 89 
122 74 95 155 
67 40 69 48
```
Se observa que el proceso 0 empieza generando dos matrices A y B, ambas de tamaño 4x4:
$
  A = mat(5, 6, 5, 8; 1, 5, 4, 2; 9, 0, 9, 4; 0, 1, 0, 8)
  " "
  B = mat(7, 2, 0, 7; 3, 0, 5, 8; 3, 4, 7, 8; 8, 5, 8, 5)
$

Posteriormente, el proceso 0 reparte las filas de la matriz A entre los procesos disponibles (0, 1, 2 y 3) mediante la función MPI_Scatter. Cada proceso recibe una fila de la matriz A y calcula su correspondiente fila de la matriz resultado C multiplicando la fila recibida con la matriz B. Finalmente, cada proceso imprime su resultado parcial y el proceso 0 imprime la matriz resultado completa.

Al finalizar, se imprime la matriz resultante C, la cual coincide con el resultado esperado de la multiplicación de las matrices A y B.

En una segunda ejecución del programa, se obtiene la siguiente salida:
```txt
Matriz A generada por proceso 0:
6 4 8 9 
6 4 6 8 
2 1 5 3 
0 8 5 3 

Matriz B generada por proceso 0:
4 3 4 5 
1 0 6 7 
9 4 1 4 
3 9 9 9 

Proceso 0 - Fila recibida de A: 6 4 8 9

Proceso 1 - Fila recibida de A: 6 4 6 8

Proceso 2 - Fila recibida de A: 2 1 5 3

Proceso 3 - Fila recibida de A: 0 8 5 3

Proceso 0 - Resultado parcial (fila de C): 127 131 137 171

Proceso 1 - Resultado parcial (fila de C): 106 114 126 154

Proceso 2 - Resultado parcial (fila de C): 63 53 46 64

Proceso 3 - Resultado parcial (fila de C): 62 47 80 103


Matriz resultado C:
127 131 137 171 
106 114 126 154 
63 53 46 64 
62 47 80 103 
```

Indicando que la generación de las matrices en el proceso 0 es aleatoria.

=== Analisis de la salida del programa de Montecarlo π
En la terminal, se posicionó sobre el directorio `./desafios_ml` y se ejecutó
```bash
mpic++ montecarlo_pi_mpi.cpp -o montecarlo_cpp && mpirun -np 4 ./montecarlo_pi_cpp
```

Para compilar y ejecutar el programa de Montercarlo para aproximar el valor de $pi$ con 4 procesos. Su salida fue la siguiente:
```txt
Proceso 0 aciertos: 196482
Proceso 3 aciertos: 196492
Proceso 2 aciertos: 196005
Proceso 1 aciertos: 196112

=== ESTIMACIÓN DE π ===
Aciertos totales: 785091
π ≈ 3.14036
Error: -0.00122865
```
En la salida, pueden verse 4 procesos identificados con numeros del 0 al 3. Cada uno muestra la cantidad de aciertos o puntos que cayeron en el circulo.

Posteriormente, se imprime la cantidad de acierto totales y se compara con $pi$ mediante el calculo:
$
  pi approx 4 dot "puntos_dentro_del_circulo"/"cantidad_total_de_puntos"
$
El cual, para esta ejecución da 3.14036 con un error de aproximadamente -0.00122865 respecto al verdadero valor de $pi$

Despejando la formual, se puede estimar que la cantidad de intentos fue:
$
  "cantidad_total_de_puntos" = (4 dot "puntos_dentro_del_circulo")/pi approx 1.000.001
$

El aparente desorden en la impresión de los aciertos de los procesos: 0,3,2,1 evidencia que los procesos corrieron en paralelo de forma asíncrona y compitieron por escribir en la consola (condición de carrera de salida), llegando al canal de texto según el orden en que finalizaron.

En una segunda ejecución, la salida fue:
```txt
Proceso 3 aciertos: 196582
Proceso 1 aciertos: 196293
Proceso 2 aciertos: 196549
Proceso 0 aciertos: 196021

=== ESTIMACIÓN DE π ===
Aciertos totales: 785445
π ≈ 3.14178
Error: 0.000187346
```
Respaldando la afirmación anterior y demostrando la aleatoridad del resultado.

=== Analisis del Algoritmo de K-means

```txt
Puntos: 1000, Procesos: 4
Centros iniciales:
C0: (5.85482, 5.85328)
C1: (9.63365, 7.41133)
C2: (9.8201, 1.25114)

✅ Centros finales tras 20 iteraciones:
Cluster 0: (2.23559, 7.08114)  -  306 puntos
Cluster 1: (7.73923, 6.81247)  -  314 puntos
Cluster 2: (4.75705, 1.96342)  -  380 puntos
```

== Respuestas a pregunas

== Conclución