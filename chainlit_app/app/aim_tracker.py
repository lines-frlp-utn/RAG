from app.config import conf


def start_aim_run():
    if conf.USE_AIM:
        from aim import Run, Text

        try:
            aim_run = Run(repo="aim://aim-server:53800", experiment="Lines chat")
            return aim_run
        except Exception as e:
            print(
                "Error starting AIM run. AIM server might not be running. USE_AIM is set to True."
            )
            print(f"Please check your AIM server connection. Error: {e}")
            return None
    else:
        return None


def end_aim_run(aim_run):
    if conf.USE_AIM:
        aim_run.close()


def track_param(aim_run, name, value):
    if conf.USE_AIM:
        aim_run.set(name, value)


def track_text(aim_run, name, text):
    if conf.USE_AIM:
        aim_text = Text(text)
        aim_run.track(aim_text, name=name)
