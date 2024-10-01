from aim import Run, Text

def start_aim_run(): 
    aim_run = Run(repo="aim://localhost:53800",
    experiment="Lines chat")
    return aim_run


def end_aim_run(aim_run):
    aim_run.close()

def track_param(aim_run,name,value):
    aim_run.set(name,value)


def track_text(aim_run,name,text):
    aim_text = Text(text)
    aim_run.track(aim_text,name=name)