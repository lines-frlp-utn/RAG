from langchain_community.callbacks import AimCallbackHandler
from langchain_core.callbacks import StdOutCallbackHandler

aim_callback = AimCallbackHandler(
    repo="aim://aim-server:53800",
    experiment_name="Lines chat",
)

callbacks = [StdOutCallbackHandler(), aim_callback]
