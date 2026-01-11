import mlflow
import mlflow.pyfunc
import pandas as pd

class SentenceTransformerPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(context.artifacts["model_dir"])

    def predict(self, context, model_input):
        # 支持 DataFrame 或 list[str]
        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].astype(str).tolist()
            else:
                texts = model_input.iloc[:, 0].astype(str).tolist()
        else:
            texts = list(model_input)

        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.tolist()


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mailru-rag")

    with mlflow.start_run(run_name="log_and_register_embeddings") as run:
        model_dir = "models/fine_tuned_embeddings"

        # 1) 作为“真正的 MLflow Model”记录（会生成 MLmodel 文件）
        mlflow.pyfunc.log_model(
            artifact_path="st_pyfunc_model",
            python_model=SentenceTransformerPyFunc(),
            artifacts={"model_dir": model_dir},
            pip_requirements=[
                "mlflow==3.8.1",
                "sentence-transformers",
                "torch",
                "numpy",
                "pandas",
            ],
        )

        # 2) 注册到 Model Registry（这样 Models 页面就会出现）
        model_uri = f"runs:/{run.info.run_id}/st_pyfunc_model"
        mlflow.register_model(model_uri=model_uri, name="mailru-rag-embeddings")

        print("RUN_ID:", run.info.run_id)
        print("MODEL_URI:", model_uri)


if __name__ == "__main__":
    main()
