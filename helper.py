import boto3
import sagemaker
import os, sys, time
from pprint import pprint
from datetime import datetime, date, timedelta
from bokeh.io import push_notebook,output_notebook, show
from bokeh.plotting import figure
from bokeh.models.widgets import Select, DatePicker, MultiSelect
from bokeh.layouts import widgetbox, column, row
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
import numpy as np
import pandas as pd
import json
from random import randint
  

region = boto3.Session().region_name
sage_client = boto3.Session().client('sagemaker')
s3 = boto3.resource('s3')

def training_status(job_name):

    client = boto3.client(service_name='sagemaker')
    status = None
    time.sleep(randint(0,10)) 
    while status is None or status == 'InProgress':
        status_obj = client.describe_training_job(TrainingJobName=job_name) 
        status = status_obj['TrainingJobStatus']
        substatus = status_obj['SecondaryStatus']

        sys.stdout.write("Training job {} status: {} - {}               \r".format(job_name, status, substatus))
        sys.stdout.flush()

        time.sleep(10)

def visualize_detection(img_file, dets, classes=[], thresh=0.6, bboxes=[]):
        """
        visualize detections in one image
        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import random
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img=mpimg.imread(img_file)
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for b in bboxes:
            rect = plt.Rectangle((b[0], b[1]), b[2], b[3], fill=False, edgecolor=(1,1,1), linewidth=3.5)
            plt.gca().add_patch(rect)
            
        for det in dets:
            (klass, score, x0, y0, x1, y1) = det
            if score < thresh:
                continue
            cls_id = int(klass)
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=3.5)
            plt.gca().add_patch(rect)
                
            class_name = str(cls_id)
            if classes and len(classes) > cls_id:
                class_name = classes[cls_id]
            plt.gca().text(xmin, ymin - 2,
                            '{:s} {:.3f}'.format(class_name, score),
                            bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=12, color='white')
        plt.show()
        
class HoverHelper():

    def __init__(self, tuning_analytics):
        self.tuner = tuning_analytics

    def hovertool(self):
        tooltips = [
            ("FinalObjectiveValue", "@FinalObjectiveValue"),
            ("TrainingJobName", "@TrainingJobName"),
        ]
        for k in self.tuner.tuning_ranges.keys():
            tooltips.append( (k, "@{%s}" % k) )

        ht = HoverTool(tooltips=tooltips)
        return ht

    def tools(self, standard_tools='pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset'):
        return [self.hovertool(), standard_tools]

def monitor_tuner(tuning_job_name):
    
    # run this cell to check current status of hyperparameter tuning job
    tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

    status = tuning_job_result['HyperParameterTuningJobStatus']
    if status != 'Completed':
        print('Reminder: the tuning job has not been completed.')

    job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
    print("%d training jobs have completed" % job_count)

    is_minimize = (tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['Type'] != 'Maximize')
    objective_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']

    if tuning_job_result.get('BestTrainingJob',None):
        print("Best model found so far:")
        pprint(tuning_job_result['BestTrainingJob'])
    else:
        print("No training jobs have reported results yet.")

def get_tuner_results(tuning_job_name):
    tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)

    full_df = tuner.dataframe()

    if len(full_df) > 0:
        df = full_df[full_df['FinalObjectiveValue'] > -float('inf')]
        if len(df) > 0:
            df = df.sort_values('FinalObjectiveValue', ascending=is_minimize)
            print("Number of training jobs with valid objective: %d" % len(df))
            print({"lowest":min(df['FinalObjectiveValue']),"highest": max(df['FinalObjectiveValue'])})
            pd.set_option('display.max_colwidth', -1)  # Don't truncate TrainingJobName        
        else:
            print("No training jobs have reported valid results yet.")

    return df

def show_tuner_eval(tuning_job_name):
    tuner = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
        
    hover = HoverHelper(tuner)

    p = figure(plot_width=900, plot_height=400, tools=hover.tools(), x_axis_type='datetime')
    p.circle(source=df, x='TrainingStartTime', y='FinalObjectiveValue')
    show(p)
    

class DeepARPredictor(sagemaker.predictor.RealTimePredictor):
        
    def predict(self, data, quantiles, encoding='utf-8', num_samples=100):
        req = self.__encode_request(data, encoding, num_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, encoding)
    
    def __encode_request(self, data, encoding, num_samples, quantiles):
        request = {
            "instances": data, 
            "configuration": {
                "num_samples": num_samples,
                "output_types": ["quantiles"],
                "quantiles": quantiles
            }
        }
        return json.dumps(request).encode(encoding)
    
    def __decode_response(self, response, encoding):
        response_data = json.loads(response.decode(encoding))
        response = [pred['quantiles'] for pred in response_data['predictions']]
        return response
    
def predict_from_file(sagemaker_session, endpoint_name, s3_bucket, path, holdout_file, context_len, max_requests=5, quantiles=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]):
    print('downloading s3://{}/{}'.format(s3_bucket, path + holdout_file))
    s3.Object(s3_bucket, path + holdout_file).download_file(holdout_file)
    
    predictor = DeepARPredictor(
        endpoint=endpoint_name,
        sagemaker_session=sagemaker_session,
        content_type="application/json"
    )

    # split off actuals and contexts to be fed into predictor
    actuals = []
    requests = []
    count = 0
    with open(holdout_file, 'r') as jsonfile:  
        for line in jsonfile:
            data = json.loads(line)     
            actuals.append(data['target'])
            context = data['target'][:context_len]
            data['target'] = context
            requests.append(data)
            count += 1
            if count >= max_requests:
                break
            
    predicteds = predictor.predict(requests, quantiles)
    return requests, actuals, predicteds


instruments = 'audusd,eurusd,usdjpy'.split(',')
columns =  'open,close,high,low,volume'.split(',')
def get_title(request):
    return instruments[int(request['cat'][1])] + ' ' +columns[int(request['cat'][0])] + ' ' + request['start']
 
def get_model(requests, actuals, predicteds):
    legends = ['{} quantile'.format(i) for i in predicteds[0].keys()] + ['actual']
    widths = [2] * 10 + [3]
    all_colors = ['black', '#191970','#000080', '#0000CD', '#0000FF', '#4169E1', '#6495ED', '#1E90FF', '#00BFFF']

    plots = []
    plot_row = []
    index = 1
    with_legend = False
    for request, actual, predicted in zip(requests, actuals, predicteds):
        context = request['target']
        quantiles = [v for k, v in predicted.items()]
        
        context_len = len(context)
        actual_len = len(actual)
        quantile_len = len(quantiles[0])
        n_quantiles = len(quantiles)
        
        x_actual = range(0, actual_len)
        x_quantiles = range(context_len, context_len+quantile_len)
        colors = all_colors[:n_quantiles]
        colors.append('red')
        
        x_all = [x_quantiles] * n_quantiles
        x_all = x_all + [x_actual]       
        y_all = quantiles + [actual]
           
     
        # create a new plot with a title and axis labels
        plot = figure(
            title = get_title(request), 
            x_axis_label='time', y_axis_label='price', 
            width=400, height=250
        )
        for x,y,c,l,w in zip(x_all,y_all,colors,legends,widths):
            if with_legend:
                plot.line(x,y,legend=l, color=c, line_width=w)
                with_legend = False
            else:
                plot.line(x,y, color=c, line_width=w)
        
        plot_row.append(plot)
                
        if index%2 == 0: 
            plots.append(row(*plot_row, width=900))
            plot_row = []
        index+= 1
    
    if len(plot_row):
        plots.append(row(*plot_row, width=900))
        
    display = column(*plots)
    return display

def graph_predictions(requests, actuals, predicteds):
    output_notebook() 
    show(get_model(requests, actuals, predicteds))
