#%%
import os

#%%
def changePicName():
    files = os.listdir('./trainingSet')
    for i in range(len(files)):
        os.rename("./trainingSet/"+files[i], "./trainingSet/"+ str(i)+".jpg")
#%%
changePicName()