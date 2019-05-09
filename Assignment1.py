#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#Name: Fernanda Pereira Guidoti
#Course Code: SCC5830
#Year/Semester: 2019/1
#Title: Assignment 2
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


# In[2]:


import numpy as np
import imageio


# In[3]:


# Funtion to read image with imageio library
def read_image(Path_Img):
    im = imageio.imread(Path_Img)
    
    if np.ndim(im)==3:
        im=np.dot(im[...,:3], [0.299, 0.587, 0.144])
        im=im.astype(np.uint8)
        
    return im


# In[4]:


#Normalizar 0 and 255
def Normalizar_255(ImagenEntrada):
    Min=np.min(ImagenEntrada)
        
    I=ImagenEntrada-Min
    
    Max=np.max(I)
    
    I=255*(I/Max)
    
    I=I.astype(np.uint8)
    
    return(I)


# In[5]:


# Limiarization function
def limiarization(Imagen,T0):
    Dif=100 # Initial value of the difference between Ti and Tj
    I=Imagen.astype(np.float32)
    Tj=T0
    while Dif>=0.5:
        Ti=Tj
        G1=np.zeros(I.shape)
        G2=np.zeros(I.shape)
        
        G1[I>Ti]=1
        G2[I<=Ti]=1

        NG1=np.sum(G1)
        NG2=np.sum(G2)

        GG1=G1*I
        GG2=G2*I
        
        # Calculate the arithmetic mean
        UG1=np.sum(GG1)/NG1
        UG2=np.sum(GG2)/NG2
        
        Tj=0.5*(UG1+UG2)
        
        Dif=np.abs(Tj-Ti)
        
        
    G1=np.zeros(I.shape)
    G1[I>Tj]=255
    
    Res=Normalizar_255(G1)
    return(Res)


# In[6]:


# 1D Filter Function
def Filtering_1D(Imagen,W,N):
    
    I=Imagen.astype(np.float32)
    L=np.reshape(I,(I.size,)) # Linearizes the image, makes it a vector
    
    NumElem=int((N-1)/2)
    
    Primero=L[0:NumElem]
    Ultimo=L[I.size-NumElem:I.size]
    
    Lplus=np.concatenate((Ultimo,L,Primero),axis=0)
    Res=np.zeros(I.size,dtype='float32')
    
    for i in range(I.size):
        S=np.reshape(Lplus[i:i+N],(1,N))
        Res[i]=np.sum(S*W)

    In=np.reshape(Res,(I.shape))
    
    In=Normalizar_255(In)
    
    return(In)


# In[7]:


# Function to 2D Filtering
def Filtering_2D(Imagen,W,N,T):
    
    I=Imagen.astype(np.float32)
    
    #Number of Rows and Columns of Image I
    m,n=I.shape
    
    #Calculate the number of elements that the Image should be extended
    NumElem=int((N-1)/2)
    
    # Create a copy of the image
#    IP=np.copy(I)
    
#    # Expand rows at the beginning and end of the image
#    Fila1=np.reshape(I[0,:],(1,n))
#    FilaN=np.reshape(I[m-1,:],(1,n))
#    for i in range(NumElem):
#        IP=np.concatenate((Fila1,IP,FilaN),axis=0)
#       
#    # Expand columns at the beginning and end of the image
#    Col1=np.reshape(IP[:,0],(IP.shape[0],1))
#    ColN=np.reshape(IP[:,IP.shape[1]-1],(IP.shape[0],1))
#    for i in range(NumElem):
#        IP=np.concatenate((Col1,IP,ColN),axis=1)
        
    IP=np.pad(I,NumElem,'constant', constant_values=(0))
    
    IF=np.zeros(I.shape,dtype=np.float32)
    for i in range(m):
        for j in range(n):
            M=IP[i:i+N,j:j+N]*W
            IF[i,j]=np.sum(M.flatten())
    
    # Normalize the imagen
    #IF=Normalizar_255(IF)
    
    IF=limiarization(IF,T)
    
    return(IF)


# In[8]:


# Function to 2D Median Filter
def Median_Filter_2D(Imagen,N):
    
    I=Imagen.astype(np.float32)
    #Number of Rows and Columns of Image I
    m,n=I.shape
    
    #Calculate the number of elements that the Image should be extended
    NumElem=int((N-1)/2)
    
    # Create a copy of the image
    IP=np.copy(I)
    
    # Expand rows at the beginning and end of the image
    Fila=np.zeros((1,n),dtype=np.float32)
    for i in range(NumElem):
        IP=np.concatenate((Fila,IP,Fila),axis=0)
       
    # Expand rows at the beginning and end of the image
    Col=np.zeros((IP.shape[0],1))
    for i in range(NumElem):
        IP=np.concatenate((Col,IP,Col),axis=1)
        
    
    IF=np.zeros(I.shape,dtype=np.float32)
    for i in range(m):
        for j in range(n):
            IF[i,j]=np.sort(IP[i:i+N,j:j+N].ravel(),axis=0,kind='heapsort')[int(N*N/2)]
            #IF[i,j]=np.median(IP[i:i+N,j:j+N], overwrite_input=True)

            
    # Normalize the imagen
    IF=Normalizar_255(IF)
    
    return(IF)


# In[ ]:


#Function to calculate root mean square error
def Error_RMSE(R,F):
    M,N=R.shape
    
    I1=R.astype(np.float32)
    I0=F.astype(np.float32)
    
    Error=np.sqrt((1/(M*N))*np.sum((I1-I0)**2))
    return(Error)
    


# In[ ]:


# Main program

NombreImagen = str(input()).rstrip()
FuncionUsada=int(input()) # Determine function

#  Load the image
I0=read_image(NombreImagen)

#According to the selected function, the image f is calculated
if FuncionUsada==1:
    T=int(input()) #Read the Threshold value
    I1=limiarization(I0,T)
elif FuncionUsada==2:
    N=int(input()) # Read the number of vectors of the vector
    W=np.reshape(np.array(list(map(float, input().split()))),(1,N))
    I1=Filtering_1D(I0,W,N)
elif FuncionUsada==3:
    N=int(input()) # Read the number of vectors of the vector
    W=np.reshape(np.array(list(map(float, input().split()))),(1,N))
    for i in range(N-1):
        Temp=np.reshape(np.array(list(map(float, input().split()))),(1,N))
        W=np.concatenate((W,Temp),axis=0)
    T=int(input()) # Read the Threshold value
    I1=Filtering_2D(I0,W,N,T)
elif FuncionUsada==4:
    N=int(input()) # Read the number of vectors of the vector
    I1=Median_Filter_2D(I0,N)


 
#Calcule Error RMSE  for images I0 and I1
print(Error_RMSE(I0,I1))

