from langchain_community.callbacks import AimCallbackHandler
from langchain_core.callbacks import StdOutCallbackHandler
from aim import Run, Text

aim_callback = AimCallbackHandler(
    repo="aim://aim-server:53800",
    experiment_name="Lines chat",
)

callbacks = [StdOutCallbackHandler(), aim_callback]

def start_aim_run(): 
    return Run(repo="aim://aim-server:53800",
    experiment_name="Lines chat"),


def end_aim_run(aim_run):
    aim_run.close()

def track_param(aim_run,name,value):
    aim_run[name]= value


def track_text(aim_run,name,text):
    aim_text = Text(text)
    aim_run.track(aim_text,name=name)