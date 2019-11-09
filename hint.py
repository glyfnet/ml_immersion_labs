import ipywidgets as widgets
from IPython.display import display
from IPython.display import Code
from IPython.display import HTML

def hint(section):
    if section in SECTION:
        _hint(SECTION[section])
        
def _hint(code):
    code_blocks = [] 
    for block in code.split('\n'):
        block = block.rstrip()
        if len(block) != 0:
            code_blocks.append(block)

    button = widgets.Button(description="Hint")
    button.count = 0
    output = widgets.Output()
    display(button, output)

    def on_button_clicked(b):
        with output:
            if button.count < len(code_blocks):
                display(Code(data=code_blocks[button.count], language='python'))
                button.count+=1
            else:
                button.disabled = True
            
    button.on_click(on_button_clicked)

SECTION = {
'estimator':
"""
estimator = sagemaker.estimator.Estimator(
    training_image,
    role,
    train_instance_count=1,
    train_instance_type='ml.p2.xlarge',
    train_max_run=10000,
    input_mode='Pipe',
    output_path=model_path,
    sagemaker_session = session
)                                                  
""",
'fit':
 """
 estimator.fit(
    inputs=data_channels, 
    job_name=job_name,
    wait=False, 
    logs=True
)
 """,
'hosting':
 """
object_detector = estimator.deploy(
    initial_instance_count = 1, 
    instance_type = 'ml.m4.xlarge'
)
""",
'infer':
"""
        results = object_detector.predict(b)
        detections = json.loads(results)
"""
}
  