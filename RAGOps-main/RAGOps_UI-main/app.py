from flask import Flask, render_template, request, jsonify 
from flask_cors import CORS
import base64
import json
from io import BytesIO
import os
import pandas as pd
from src.mlflow_data import *
from src.gitconnect import *
from src import pipeline
import shutil
#####################################
# GLOBAL VARIABLES AND DECLARATIONS #
# ####################################


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'GET'])
def bank():
    # Experiments tab - Run/Commit section
    # Create a dictionary containing all the details of run wise details for Runs/Commit Tab
    run_details_all_dict_bank = RUN_WISE_DATA_BANK_MLFLOW.to_dict('records')
    #Setting the current commit ID variable to latest when the bank page loads from global variable
    #current_run_id_bank = DEPLOYED_RUN_ID_MODEL_BANK
    return render_template('bank.html',run_details_all_dict_bank = run_details_all_dict_bank)
                           #,current_run_id_bank = current_run_id_bank)
    

@app.route('/get_runs_experiments_deploy', methods = ['GET', 'POST']) 
def get_params_commits_bank_mlfow():
    # get the selected column from the drop-down list
    selected_column = request.args.get('selected_value')
    print(f"Selected sorting based on feature column from the Ajax request ------> {selected_column}")
 
    print("###################", RUN_WISE_DATA_BANK_MLFLOW)
    # sort the DataFrame based on the selected column
    if selected_column == "Latest":
        run_metric_df_bank = RUN_WISE_DATA_BANK_MLFLOW
        print("***************** SORT ON THE LATEST ********", run_metric_df_bank )
    elif selected_column == "faithfulness":
        run_metric_df_bank = RUN_WISE_DATA_BANK_MLFLOW.sort_values(by="faithfulness", ascending=False)
        print(" &&&&&&&&&&&&&& SORT ON faithfulness &&&&&&&&&&&", run_metric_df_bank)
    elif selected_column == "relevancy":
        run_metric_df_bank = RUN_WISE_DATA_BANK_MLFLOW.sort_values(by="relevancy", ascending=False)
        print(" &&&&&&&&&&&&&& SORT ON relevancy &&&&&&&&&&&", run_metric_df_bank)
    
    # convert the sorted DataFrame to a list of dictionaries
    run_details_all_dict_bank = run_metric_df_bank.to_dict('records')
    # store the commit _id which is default from the global variable and pass to ajax call
    #current_run_id_bank = DEPLOYED_RUN_ID_MODEL_BANK
    #print(f"Current commit ID model being deployed for Inference is : {current_run_id_bank}")
    
    # return the sorted DataFrame and deployed commit ID as a JSON response
    return jsonify(run_details_all_dict_bank=run_details_all_dict_bank)#, current_run_id_bank=current_run_id_bank)

            
@app.route('/set_deployed_run_id_bank', methods = ['GET','POST']) 
def set_deployed_run_id_bank_mlfow():
    #global DEPLOYED_RUN_ID_MODEL_BANK
    data = request.get_json()
    DEPLOYED_RUN_ID_MODEL_BANK = data['deployed_run_id']
    deploydata = RUN_WISE_DATA_BANK_MLFLOW[RUN_WISE_DATA_BANK_MLFLOW['Run_id'] == DEPLOYED_RUN_ID_MODEL_BANK]
    deploydata = deploydata.to_dict('records')
    print('############ deploy data ##########')
    print(deploydata[0])
    pipeline.deploy_rag(deploydata[0])
    print(f"deployed model from run ID {DEPLOYED_RUN_ID_MODEL_BANK}") 
    return jsonify({'current_run_id_bank': DEPLOYED_RUN_ID_MODEL_BANK})

@app.route('/configure_data', methods=['POST'])
def configure_data():
    datafiles = request.files.getlist('data-rag')
    userdata = request.form.to_dict()
    print(userdata)
    # Data to be written
    if userdata['type_granularity'] == 'chunk':
        data = {
            "collection": userdata['collection'],
            "type_granularity": userdata['type_granularity'],
            "chunk_size": int(userdata['chunk_size']),
            "Embeddings": userdata['Embeddings'],
        }
    else:
        data = {
            "collection": userdata['collection'],
            "type_granularity": userdata['type_granularity'],
            "sentence_window_size": int(userdata['sentence_window_size']),
            "Embeddings": userdata['Embeddings'],
        }
    # Serializing json
    json_object = json.dumps(data, indent=3)

    with open('config/data.json', "w") as outfile:
        outfile.write(json_object)
        
    collection = userdata['collection']
    path = f'/workspace/RAGOps_artifacts/collections/{collection}'
    if not os.path.exists(path):
        os.makedirs(path)
    for file in datafiles:
        filename = file.filename
        if filename == '':
            break
        temp_path = os.path.join(path,filename)
        file.save(temp_path)
    
    print('^^^^^^^^^^ Data pipeline configured ^^^^^^^^^^^')
    return jsonify({'data_config_msg':'Data pipeline configured!'})

@app.route('/configure_rag', methods=['POST'])
def configure_rag():
    userdata = request.form.to_dict()
    print(userdata)

    # Data to be written
    data = {
        "similarity_top_k": int(userdata['similarity_top_k']),
        "rerank_top_n": int(userdata['rerank_top_n']),
        "LLM": userdata['LLM'],
        "temperature": float(userdata['temperature']),
        "context_window": int(userdata['context_window']),
        "max_new_tokens": int(userdata['max_new_tokens']),
    }
    # Serializing json
    json_object = json.dumps(data, indent=3)

    with open('config/rag.json', "w") as outfile:
        outfile.write(json_object)

    return jsonify({'rag_config_msg':'Retriever and generator configured!'})

@app.route('/trigger_experiment', methods=['POST'])
def trigger_experiment():
    ci_trigger()
    return jsonify({'trigger_msg':'Experiment Triggered!'})

@app.route('/query_rag', methods=['POST'])
def query_rag():
    userdata = request.form.to_dict()
    print(userdata)
    response = pipeline.run_query(userdata)
    return jsonify({'ans_rag':response})


if __name__ == '__main__':

    df = extract_runs_mlflow_bank( mlflow_tracking_uri = "http://10.161.2.132:5020", Experiment_name = 'RAGOps')

    RUN_WISE_DATA_BANK_MLFLOW = df.head(10)
    
    #run_wise_data_bank_mlflow_sorted = RUN_WISE_DATA_BANK_MLFLOW.sort_values(by="faithfulness", ascending=False)
    #run_wise_data_bank_mlflow_sorted = RUN_WISE_DATA_BANK_MLFLOW
    #DEPLOYED_RUN_ID_MODEL_BANK = run_wise_data_bank_mlflow_sorted['Run_id'].tolist()[0]
    
    print('http://10.161.2.132:5008')

    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)
