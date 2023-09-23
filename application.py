from flask import Flask,render_template,request
from src.pipes.prediction_pipe import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="GET":
        return render_template("form.html")
    
    else:
        Data=CustomData(
            Cement=float(request.form.get("Cement"))
            Blast_furnace_slag=float(request.form.get("Blast_furnace_slag"))
            Fly_ash=float(request.form.get("Fly_ash"))
            Water=float(request.form.get("Water"))
            Superplasticizer=float(request.form.get("Superplasticizer"))
            Coarse_aggregate=float(request.form.get("Coarse_aggregate"))
            Fine_aggregate=float(request.form.get("Fine_aggregate"))
            Age=float(request.form.get("Age"))
        )

        final_data=Data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        predict=predict_pipeline.predict(final_data)

        result=round(pred[0],3)

        return render_template("form.html",final_results=result)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
