import joblib
import pandas as pd
pipe_loaded = joblib.load("./Codes/logreg_pipeline.joblib")
"""title_abstract	authkeywords	subject-area	sourcetitle	srctype	openaccess	year"""
x = [{"title_abstract":"blablabla", "authkeywords":None,	"subject-area" : "Medicine (all)", "sourcetitle": "", "srctype":"j", "openaccess":1,	"year":2018}]
x = pd.DataFrame(data = x)
print(x)
def predictWithThreshold(model , x , threshold = 0.55):
    y_proba = model.predict_proba(x)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred

print(predictWithThreshold(pipe_loaded,x))