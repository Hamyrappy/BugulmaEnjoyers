from bugulma_enjoyers.models import MODEL_TYPES


def load_model(model_name, pipeline_config, **kwargs):
    splits = model_name.split("/")
    model_type = splits[0]
    model_name = "/".join(splits[1:])

    return MODEL_TYPES[model_type](model_name=model_name, pipeline_config=pipeline_config, **kwargs)
