from datetime import datetime
import pandas as pd
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient

import warnings
warnings.filterwarnings('ignore')


def extract_runs_mlflow_bank(mlflow_tracking_uri, Experiment_name):
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)
    
    Exp_details = client.get_experiment_by_name(Experiment_name)
    Exp_id = Exp_details.experiment_id
    print(f"Experiment id: {Exp_id}")

    runs_list = client.search_runs(
                                    experiment_ids=Exp_id,
                                    filter_string='attributes.status = "FINISHED"',
                                )
    runs_df = pd.DataFrame()
    for run in runs_list:
 
        run_dict = {}
        run_dict["Run_id"] = run.info.run_id
        run_dict["Run_Name"] = run.info.run_name
        #run_dict["Atrifact_URI"] = run.data.tags['Atrifact_URI']
        run_dict["Eval-LLM"] = run.data.tags['Eval-LLM']
        start_time = (run.info.start_time)
        # created_date_str = datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M:%S')
        # run_dict["Created"] = datetime.strptime(created_date_str, '%Y-%m-%d %H:%M:%S')
        run_dict["Created"] = created_date_str = datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M:%S')
        run_dict["Duration(MM:SS)"] = pd.to_datetime(((run.info.end_time - run.info.start_time)), unit='ms').strftime("%M:%S")

        metrics = run.data.metrics
        for metric in metrics:
            run_dict[metric] = round(metrics[metric], 3)

        hyperparmeters = run.data.params
        for param in hyperparmeters:
            run_dict[param] = hyperparmeters[param]

        run_dict_df = pd.DataFrame.from_dict([run_dict])
        
        runs_df = pd.concat([runs_df, run_dict_df], axis=0)

        columns_to_exclude = ['Run_Name','Run_id','Created',"Duration (MM:SS)"]
        subset_runs_cols = [i for i in runs_df if i not in columns_to_exclude]
        runs_df.drop_duplicates(subset=subset_runs_cols, keep=False, inplace=True)

        runs_df= runs_df.fillna('-')
        #runs_df.to_csv("runs_list_mlflow.csv", index=False)
    
    return runs_df

#if __name__ == '__main__':
#     extract_runs_mlflow_bank( mlflow_tracking_uri = "http://10.161.2.132:5020",
#                         Experiment_name = 'RAGOps')