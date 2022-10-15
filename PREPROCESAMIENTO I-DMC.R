#################################
# CURSO: MINERÍA DE DATOS CON R #
#        (NIVEL BÁSICO)         #
#  PRE-PROCESAMIENTO DE DATOS   #
#        (PRIMERA PARTE)        #
#   Mg. Jesús Salinas Flores    # 
#    jsalinas@lamolina.edu.pe   #
#################################

#---------------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls())

#---------------------------------------------------------------
# Para limpiar el área de gráficos
dev.off()
dev.new() #esto es para que se abra una nueva ventana de grafico
#---------------------------------------------------------------
# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


#---------------------------------------------------------------
# Limpiar la consola
cat("\014")
options(scipen=999)  #eliminar la notación científica
options(digits = 3)  #número de decimales

#---------------------------------------------------------------
# Paquetes
library(DataExplorer)
library(VIM)
library(naniar)
library(caret)
library(cowplot)
library(missForest)
library(dummies)
library(tictoc)
library(data.table)
library(DMwR)
library(arules)
library(funModeling)
library(ggplot2)
library(plotly)
library(dplyr)


#################################
# I. CASO DREAM HOUSING FINANCE #
#################################

#############################
#  1. Descripción del caso  #
#############################

# Introducción:
# La compañía Dream Housing Finance se ocupa de todos los 
# préstamos hipotecarios. Tiene presencia en todas las áreas 
# urbanas, semi urbanas y rurales. 
# El cliente primero solicita un préstamo hipotecario y luego  
# la compañía valida la elegibilidad del cliente para el 
# préstamo.
#
# Problema:
# La empresa desea automatizar el proceso de elegibilidad del 
# préstamo (en tiempo real) en función del detalle del cliente
# que proporcionó al completar el formulario de solicitud. 
# Estos detalles son:
# Género, Estado civil, Educación, Número de dependientes, 
# Ingresos, Monto del préstamo, Historial de crédito y otros. 
# 
# Para automatizar este proceso, se desea identificar los 
# grupos de clientes, que son elegibles para el monto del 
# préstamo para que puedan dirigirse específicamente a estos 
# clientes. 

# Lectura de los datos 
datos<-read.csv("loan_prediction-II.csv",
                stringsAsFactors = T, 
                sep=";",
                na.strings = "")
#mira la diferencia con el de arriba, este pone null tambien
#datos1<-read.csv("loan_prediction-II.csv",
#                stringsAsFactors = T, 
#                sep=";")
            
# Viendo la estructura de los datos
str(datos)

# Eliminando la columna de identificación del cliente (Loan_ID)
datos$Loan_ID <- NULL

# Declarar la variable Credit_History como factor
datos$Credit_History          <- as.factor(datos$Credit_History)
levels(datos$Credit_History)  <- c("Malo","Bueno")

# Evaluando la variable target Loan_Status
table(datos$Loan_Status)

prop.table(table(datos$Loan_Status))

str(datos)

##################################
# 2. Detección de datos perdidos #
##################################

#---------------------------------------------------------------
# Exploración de datos perdidos con DataExplorer y VIM

library(DataExplorer)
plot_missing(datos)

# Para ver cuantas filas tienen valores perdidos
rmiss <- which(rowSums(is.na(datos))!=0,arr.ind=F)
length(rmiss)

# Para ver el porcentaje de filas con valores perdidos
length(rmiss)*100/dim(datos)[1]

# Total de datos perdidos
sum(is.na(datos))   

# ¿Por qué la diferencia entre el total de datos perdidos
#  y el número de filas con datos perdidos?


# Graficar la cantidad de valores perdidos
library(VIM)
graf_perdidos1 <- aggr(datos,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)

matrixplot(datos,
           main="Matrix Plot con Valores Perdidos",
           cex.axis = 0.6,
           ylab = "registro")

#---------------------------------------------------------------
# Exploración de datos perdidos con naniar
library(naniar) 

# Total de datos perdidos
n_miss(datos)      
prop_miss(datos)   # 2831/(11500*13)

# Datos perdidos por variable
miss_var_summary(datos)

n_miss(datos$Gender)
n_miss(datos$Loan_Status)

# Visualizando los datos perdidos por casos y variables
vis_miss(datos)

vis_miss(datos, cluster = TRUE) #este es el más importante

#---------------------------------------------------------------
# Consideraciones con las variables con datos missing
# Variables Categóricas
table(datos$Gender)
addmargins(table(datos$Gender))

table(datos$Gender, useNA = "always")
addmargins(table(datos$Gender,useNA = "always"))

# Variables Numéricas
sum(datos$LoanAmount)
sum(datos$LoanAmount,na.rm=T)


#################################
# 3. Pre-procesamiento de datos #
#################################

#---------------------------------------------------------------
# Opción 1
# Imputando los valores perdidos cuantitativos usando k-nn
# y estandarizando las variables numéricas (por defecto)
library(caret)
preProcValues1 <- preProcess(datos,
                             method=c("knnImpute"))

# Asumiendo que no hay datos perdidos
# preProcValues1 <- preProcess(datos,
#                              method=c("center", "scale"))
#                              method=c("range"))
# Otras opciones: range , bagImpute, medianImpute, pca
#                 k= 3
# (X-MEDIA) / DESVIACIÓN ESTÁNDAR
preProcValues1

datos_transformado1 <- predict(preProcValues1, datos)
plot_missing(datos_transformado1)

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado1))

# Graficar la cantidad de valores perdidos en las 
# variables categóricas
graf_perdidos2 <- aggr(datos_transformado1,prop = F, 
                       numbers = TRUE,
                       sortVars=T,
                       cex.axis=0.5)


# Imputar de valores categóricos usando el algoritmo Random Forest
library(missForest)
set.seed(123)
impu_cate           <- missForest(datos_transformado1)
datos_transformado1 <- impu_cate$ximp

# Verificando la cantidad de valores perdidos
sum(is.na(datos_transformado1))

plot_missing(datos_transformado1)

# Identificando variables con variancia cero o casi cero
library(caret)
nearZeroVar(datos_transformado1, saveMetrics= TRUE)
#nvz es poca variabilidad , zerVar es cuando los datos son iguales
#                      freqRatio  percentUnique zeroVar   nzv
# Gender                 7.85        0.0174   FALSE     FALSE
# Married                3.56        0.0174   FALSE     FALSE
# Dependents             2.96        0.0348   FALSE     FALSE
# Education              3.20        0.0174   FALSE     FALSE
# Self_Employed          8.51        0.0174   FALSE     FALSE
# ApplicantIncome        1.02        4.3913   FALSE     FALSE
# CoapplicantIncome      3.04        3.4000   FALSE     FALSE
# LoanAmount             1.09        1.8435   FALSE     FALSE
# Loan_Amount_Term      11.68        0.0957   FALSE     FALSE
# Credit_History         5.66        0.0174   FALSE     FALSE
# Property_Area          1.18        0.0261   FALSE     FALSE
# Nacionality          337.24        0.0174   FALSE     TRUE
# Loan_Status            2.51        0.0174   FALSE     FALSE

table(datos_transformado1$Nacionality) #esta variable no afecta mucho en el modelo
datos_transformado1$Nacionality <- NULL

# Verificando freqRatio y percentUnique para Gender
table(datos_transformado1$Gender)

# Female   Male 
#   1299  10201 

# freqRatio     = (10201/1299)   = 7.85  mayor valor / el segundo mayor valor
# percentUnique = (2/11500)*100  = 0.01739130   2 es el numero de categorias de la variable

# Verificando la estructura del archivo pre-procesado
str(datos_transformado1)

table(datos_transformado1$Property_Area)

#---------------------------------------------------------------
# Creando variables dummies

# Usando el paquete dummies
library(dummies)
library(fastDummies) #uso este ya que dummies es una libreria antigua
#se ponen en variables dummies todo las variables categoricas pero menor la variable target o el y.
datos_transformado1 <- dummy_cols(datos_transformado1,
                                        select_columns =c("Gender","Married","Dependents",
                                                "Education","Self_Employed",
                                                "Credit_History","Property_Area"),remove_selected_columns = TRUE)
                          
# Verificando la estructura del archivo pre-procesado
str(datos_transformado1)
datos_transformado1 <- datos_transformado1[, -c(7,9,13,15,17,19,22)]

str(datos_transformado1)


