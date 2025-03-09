
import git
import os
import yaml
import shutil
import argparse
import json

def ci_trigger():
    dir = ['temp']
    for direc in dir:
        if os.path.exists(direc):
            shutil.rmtree(direc, ignore_errors=True)
        os.mkdir(direc)
        
    repo = git.Repo.clone_from("", "temp") 
    #checkout to branch
    branch_name='dev'
    repo.git.checkout(branch_name)
    
    cmd_copy = f'cp -r config/. temp/config/'
    os.system(cmd_copy)

    path = os.getcwd()
    new_path = path+'/temp'
    os.chdir(new_path)

    # ...
    repo.git.add(update=True)
    # git.cmd.Git().add('.')
    repo.index.commit('Experiment created')
    # git.cmd.Git().commit('-m','Updated hyperparameters')
    
    #Push the changes back to github
    repo.remote().push(refspec='{}:{}'.format('dev','dev'))
    # repo.git.push('origin','dev')
    # git.cmd.Git().push('origin','dev')
    
    print('Experiment pushed successfully')
    

    
if __name__=="__main__":
    airig_trigger()
    
   
