# estudiantes-tensorflow

## Integrantes
- Bernardo Altamirano (167881)
- Eduardo Pesqueira Zorrilla (176065)
- Ian Zaidenweber (176705)
- Antonino Garcia (180164)

## Objetivo
El objetivo de este proyecto es entrenar a un modelo mediante redes neuronales para que logre predecir, con base en una serie de datos, la calificación final de un estudiante en un curso de matemáticas. 


## Requerimientos
Para entrenar exitosamente el programa se requeiere de `Python`, y una serie de librerias derivadas de este, de las cuales figuran `TensorFlow`, `Pandas`, `Sklearn` y la clase `utils`. 
Asi mismo, el programa debe recibir un archivo csv con los datos que serán estudiados.


## Manual de uso
1. Abrir el proyecto estudiantes-tensorflow en Python. 
2. Aqui se encontrarán los datos sin ser modificados. La clase `limpiar_datos` limpiará los datos para poder ser ingresados al modelo y analizados correctamente.
2. Correr el proyecto
3. En las gráfics se podrán observar y comparar los resultados y predicciones del modelo una vez entrenado con el modelo sin entrenarse.

## Validaciones
Hay algunas validaciones para que el modelo logre un correcto entrenamiento:
- Algunas de las variables estudiadas deben de convertirse en nominales
- Ciertas variables de los datos no son de interes para el modelo por lo que se eliminan para no ser consideradas en el entrenamiento.
- Se separan en distintos dataframes la variable bjetivo (`G3`) y las variables de soporte, de esta manera se podrán generar los conjuntos de test y train para cada grupo de variables. 



