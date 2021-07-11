#!/usr/bin/env python
# coding: utf-8

# In[49]:


import cv2
import numpy as np
 

detec = []
toplam= 0
giris=0
cikis=0

	
def ort_calc(x, y, w, h): # dortgenin orta noktasini buldurtma
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG() # background segmentation

while True:
    ret , frame1 = cap.read() # frame yakalama
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) # frame'i griye cevirme
    blur = cv2.GaussianBlur(grey,(3,3),5) # frame'i yumusatma
    img_sub = subtractor.apply(blur) 
    dilat = cv2.dilate(img_sub,np.ones((5,5))) # frame'i genisletme 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # kernel olusturma
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel) 
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # tespit
    
    cv2.line(frame1, (224,500), (550,500), (0,0,255), 2)# giris alt cizgisi
    cv2.line(frame1, (225,510), (550,510), (0,0,255), 2)# giris üst cizgisi
    

    cv2.line(frame1, (650, 400), (650, 1000), (255, 255, 255), 2)#yon ayirma cizgisi
    
    cv2.line(frame1, (730, 544), (1050, 544), (0,0,255), 2) # cikis üst cizgisi
    cv2.line(frame1, (730, 556), (1050, 556), (0,0,255), 2) # cikis üst cizgisi
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contour = (80 >= 10) and (h >= 80) #hareketli nesneleri algılama hassasiyeti
        if not validar_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2) # aracların etrafındaki dörtgeni cizdirme       
        ort = ort_calc(x, y, w, h) # dörtgenin ortasını hesaplama
        detec.append(ort) # listeye ekleme
        cv2.circle(frame1, ort, 4, (0, 0,255), -1) # ortaya nokta cizdirme

        for (vX,vY) in detec:
            if vY > 500 and vY < 510:  # giris hesaplama // nokta giris cizgilerinin arasındaysa            
                detec.remove((vX,vY))
                if vX< 650: # serit ayirma cizgisinin solundaysa 
                        giris+=1
                        toplam += 1
                    
            if vY > 544 and vY < 556: # cikis hesaplama // nokta cikis cizgilerinin arasindaysa               
                detec.remove((vX,vY))
                if vX> 650: # serit ayirma cizgisinin solundaysa
                    cikis+=1
                    toplam += 1
                    
       
    cv2.putText(frame1, "Giris : "+str(giris)+' Cikis: '+str(cikis)+' Toplam: '+str(toplam), (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5) # verileri yazdirma
    cv2.imshow("Video" , frame1) # videoyu gosterme
    

    if cv2.waitKey(1) == 27: # cikis icin '1' tuslama
        break
    
cv2.destroyAllWindows()
cap.release()


# In[ ]:





# In[ ]:





# In[ ]:




