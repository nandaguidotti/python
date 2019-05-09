#!/usr/bin/env python
# coding: utf-8

# In[22]:


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Name: Fernanda Pereira Guidoti
#Course Code: SCC5830
#Year/Semester: 2019/1
#Title: Assignment 1 : image generation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#library
import numpy as np
import random


# In[23]:


#Function normalize img between 0 and 65535

def Normalizar(ImagenEntrada):
    Min=np.min(ImagenEntrada)
        
    I=ImagenEntrada-Min
    
    Max=np.max(I)
    
    I=65535*(I/Max)
    return(I)


# In[24]:


#Normalizar 0 y 255
def Normalizar_255(ImagenEntrada):
    Min=np.min(ImagenEntrada)
        
    I=ImagenEntrada-Min
    
    Max=np.max(I)
    
    I=255*(I/Max)
    return(I)


# In[25]:


#Function to scale the image
#Definition of scaling function
def Escalar_Imagen(Imagen,N):
    C,C=Imagen.shape
    NuevaImagen=np.zeros((N,N))
    Paso=int(C/N)
    for i in range(N):
        x=Paso*i
        for j in range(N):
            y=Paso*j
            NuevaImagen[i,j]=Imagen[x,y]
    return(NuevaImagen)


# In[26]:


#Function for bit shifting
def ShiftingImage(Imagen,B):
    DifB=8-B    
    NuevaImagen=Imagen.astype(np.uint8)
    NuevaImagen=NuevaImagen>>DifB
    return(NuevaImagen)


# In[ ]:


#Function to calculate error
def Error_RSE(R,F):
    N,N=R.shape
    Error=0.0
    Error=np.sqrt(np.sum((F-R)**2))
    return(Error)


# In[28]:


#Function 1 f(x,y)=xy+2y
#Definition function 1

def F1(C):
    #Matrix 0 CxC
    Arreglo=np.zeros((C,C))
    
    #Fill the matrix with the values of the function using two FOR cycles
    for i in range(C):
        for j in range(C):
            #Assign to each cell the value calculated with the function f(x, y) = (xy + 2y)
            x=i
            y=j
            Arreglo[i,j] = x*y + 2*y
            
    return Arreglo

#ends funcion 1


# In[29]:


#Function 2 f(x, y) = | cos(x/Q) + 2 sin(y/Q)|;
#Definition function 2
def F2(C,Q):
    #Matrix 0 CxC
    Arreglo=np.zeros((C,C))
    
    #Fill the matrix with the values of the function using two FOR cycles
    for i in range(C):
        for j in range(C):
            #Assign to each cell the value calculated with the function f(x, y) = | cos(x/Q) + 2 sin(y/Q)|
            x=i
            y=j
            Arreglo[i,j] = np.abs(np.cos(x/Q)+2*np.sin(y/Q))

    return Arreglo

#ends function 2


# In[30]:


#Function 3 f(x, y) = |3(x/Q) − raiz_cubica(y/Q)|;
#Definition function 3
def F3(C,Q):
    #Matrix 0 CxC
    Arreglo=np.zeros((C,C))
    
    #Fill the matrix with the values of the function using two FOR cycles
    for i in range(C):
        for j in range(C):
            # Assign to each cell the value calculated with the function f(x, y) = |3(x/Q) − ((1/3)y/Q)|
            x=i
            y=j
            Arreglo[i,j] = np.abs(3*(x/Q)-np.power(y/Q,1/3))

    return Arreglo

#ends function 3            


# In[31]:


#Function 4 f(x, y) = random(0,1,S)
#Definition function 4
def F4(C,S):
    #Matrix 0 CxC
    Arreglo=np.zeros((C,C))
    random.seed(S)
    
    #Fill the matrix with the values of the function using two FOR cycles
    for i in range(C):
        for j in range(C):
            # Assign to each cell the value calculated with the function f(x, y) = random(0, 1, S)
            x=i
            y=j           
            Arreglo[i,j] = random.random()

    return Arreglo

#ends function 4            


# In[ ]:


#Function 5 f(x, y) = randomwalk(S);
#Definition function 5

def F5(C,S):
    #Matrix 0 CxC
    Arreglo=np.zeros((C,C))
    random.seed(S)
    x=0
    y=0
    #Fill the matrix with the values of the function using two FOR cycles
    for i in range((C*C)+1):
            # Assign to each cell the value calculated with the function f(x, y) = randomwalk(S)
            #Calculate dx and dy
            dx=random.randint(-1,1)
            dy=random.randint(-1,1)
            x=np.mod(x+dx,C)
            y=np.mod(y+dy,C)
            Arreglo[x,y] = 1
    #Condicion f(0,0)=1
    Arreglo[0,0]=1

    return Arreglo

#ends function 5


# In[ ]:


#reed files
filename = str(input()).rstrip()

#assign the elements of the list to the respective variables
C=int(input()) #Second item in the Lines list, convert to a whole numerical value
FuncionUsada=int(input()) #third item in the Line list, you have to convert it to a whole numerical value
Q=int(input()) #Fourth item in the Lines list, convert to whole numerical value
N=int(input()) #Fifth item in the Lines list, convert to full numerical value
B=int(input()) #Sixth item in the Lines list, convert to a whole numerical value
S=int(input()) #Sept element of the Lines list, you have to convert to a whole numerical value


# In[14]:


# load the reference image
R=np.load(filename)


# In[15]:


#According to the selected function, the image f is calculated
if FuncionUsada==1:
    G=F1(C)
elif FuncionUsada==2:
    G=F2(C,Q)
elif FuncionUsada==3:
    G=F3(C,Q)
elif FuncionUsada==4:
    G=F4(C,S)
elif FuncionUsada==5:
    G=F5(C,S)


# In[16]:


#Normalization, scaling and shifting of the image
G=Normalizar(G)
G=Escalar_Imagen(G,N)
G=Normalizar_255(G)
G=ShiftingImage(G,B)


# In[17]:


#Calcule Error RSE image between R ana G
print(Error_RSE(R,G))


# In[ ]:




