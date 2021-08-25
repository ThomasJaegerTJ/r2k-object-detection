# r2k-object-detection

## Dataset versioning, storing, sharing and retrieving

#### dvc init

Repository r2k-object-detection 
leer, außer einen Ordner data/datasets, dieser enthält ImageTagger 15 GB (ist nicht der komplette Datensatz) und ist zu finden auf der zwoogle4 /workspace/datasets.

```
dvc init
```

und commit.

#### Datensatz zu dvc hinzufügen

Dann den Datensatz zu DVC hinzufügen.

```
dvc add data/datasets/ImageTagger
```

-> das erzeugte ImageTagger.dvc

dvc verlagert den Datensatz ins Projekt Cache (.dvc/cache) und erzeugt einen Link (ImageTagger.dvc).
git tracked ab nun nur noch das ImageTagger.dvc

Sollte sich am Datensatz etwas verändern, so muss man wieder den Befehl ```git add``` ausführen und committen und pushen.

```
dvc add ImageTagger
git commit ImageTagger.dvc -m "Update Dataset"
dvc push
```

Nun haben wir den Datensatz im Cache, er muss aber in einem Remote Storage lagern. Dieser liegt auf der zwoogle4.

Dazu erstmal auf der zwoogle4 einen Ordner /home/maker/dvc-storage erstellen.
Dann remote storage zu dvc hinzufügen:
```
dvc remote add -d zwoogle4 ssh://maker@zwoogle4.ds.fh-kl.de/home/maker/dvc-storage

```
Anschließend muss man den Befehl ``` dvc push ``` eingeben. Dadurch wird das Datenset aus dem Cache in den Remote Storage geladen.

Wenn man mit einem anderen Computer an anderer Stelle ``` dvc pull ``` ausführt, dann wird das Datenset aus diesem Remote Storage geladen.


### Ein bestimmtes Datenset pullen

```
dvc pull -r <name>
```

Mit dem Befehl ```dvc remote list``` kann man sich die verschiedenen Remote Storage anzeigen lassen.

Beispiel:

```
dvc pull -r zwoogle4
```

Dieser Befehl lädt den großen Datensatz herunter.

Mit ``` dvc pull -r mini-ImageTagger``` kann man den kleineren Datensatz zum Testen herunterladen.


---

## Trainings Pipeline

Output des Befehls ```dvc dag```:

       --- -----------     
       | generate_csv |     
       +--------------+     
               *            
               *            
               *            
     +-------------------+  
     | generate_tfrecord |  
     +-------------------+  
         **         **      
       **             **    
      *                 **  
+-------+                 * 
| train |               **  
+-------+             **    
         **         **      
           **     **        
             *   *          
         +----------+       
         | evaluate |       
         ------------
		 
Die Trainingspipelin kann mit dem Befehl ```dvc repro``` gestartet werden.

##### Wichtige dvc files

**dvc.yaml** 
Hier ist die Trainings Pipeline definiert 

**config.yaml**


**params.yaml**

---
