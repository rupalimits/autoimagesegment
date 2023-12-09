# ImageCaptioning

Repository for capstone model.

# Run the project

- Change the configuration file in `./image_captioning/config.yml` for changing the variables.
- Create and activate python virtual environment if required.
- Install the requirements using below code
  
```
pip install -r requirements/requirements.txt
```
- run the pipeline using below code
```
python image_captioning/main.py
```
- metrics to be incorporated
  
```
rouge2 and bert has to be incorporated and monitoring has to be in place..
```

- training time
```
ETA - 40 hours in windows,19 hours in mac8gb
Compute - 32 gb /64gb
```

- ci/cd track -- in progress
```
git-github_actions-docker-api_deployment with ec2--promethues/grafana
```
