# Belief Classification Service  
Belief prediction flask API.   
  
## Usage
  
To run with Python>=3.7:  
  
```  
pipenv shell
pipenv install 
python app.py  
```  
  
To run with Docker:  
  
First build the `prediction_base` image locally.  

```
cd prediction_base  
docker build -t prediction_base  
```  
Then run the service using docker-compose:  
```  
cd ..
sh run_docker.sh
```  
  
The API should then become available at http://localhost:5000.  
